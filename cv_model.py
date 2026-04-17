from __future__ import annotations

import os

# Set environment variables BEFORE any imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import logging
from functools import lru_cache
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from PIL import Image
from transformers import pipeline

tf.get_logger().setLevel("ERROR")
warnings.filterwarnings(
    "ignore",
    message=r"The structure of `inputs` doesn't match the expected structure\.",
    category=UserWarning,
    module=r"keras\.src\.models\.functional",
)

XRAY_VALIDATION_MODEL_ID = "openai/clip-vit-base-patch32"
XRAY_VALIDATION_LABELS = [
    "a chest x-ray radiograph",
    "a chest radiograph",
    "a medical x-ray image",
    "a CT scan image",
    "an MRI scan image",
    "an ultrasound image",
    "a photograph of a person",
    "a screenshot",
    "a document page",
    "a natural scene",
]


@lru_cache(maxsize=1)
def _get_xray_validator():
    return pipeline(
        "zero-shot-image-classification",
        model=XRAY_VALIDATION_MODEL_ID,
        device=-1,
    )


def get_preprocess_fn(backbone: str):
    if backbone == "EfficientNetV2S":
        return tf.keras.applications.efficientnet_v2.preprocess_input
    if backbone == "ResNet50":
        return tf.keras.applications.resnet50.preprocess_input
    return tf.keras.applications.mobilenet_v2.preprocess_input


def build_transfer_model(backbone: str = "ResNet50", image_size: Tuple[int, int] = (224, 224)) -> tf.keras.Model:
    input_shape = (image_size[0], image_size[1], 3)

    if backbone == "EfficientNetV2S":
        base_model = tf.keras.applications.EfficientNetV2S(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    elif backbone == "ResNet50":
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    else:
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = get_preprocess_fn(backbone)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def load_image_datasets(
    dataset_root: Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
):
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds


def train_or_load_cv_model(
    dataset_root: Path,
    model_path: Path,
    backbone: str = "ResNet50",
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    epochs: int = 2,
    evaluate_loaded_model: bool = False,
) -> Tuple[tf.keras.Model, Dict[str, float]]:
    if model_path.exists():
        try:
            # Some environments fail to load optimizer/training metadata from older .keras files.
            model = tf.keras.models.load_model(model_path, compile=False)
            if evaluate_loaded_model and dataset_root.exists():
                try:
                    _, _, test_ds = load_image_datasets(
                        dataset_root,
                        image_size=image_size,
                        batch_size=batch_size,
                    )
                    eval_vals = model.evaluate(test_ds, verbose=0)
                    return model, {"loss": float(eval_vals[0]), "accuracy": float(eval_vals[1]), "auc": float(eval_vals[2])}
                except Exception:
                    return model, {}

            return model, {}
        except Exception as load_exc:
            # If dataset is available, retrain and replace bad/incompatible model file.
            if not dataset_root.exists():
                raise ValueError(
                    f"Saved model could not be loaded at {model_path}. "
                    "Provide a compatible model file or deploy with training dataset to retrain."
                ) from load_exc
            try:
                model_path.unlink(missing_ok=True)
            except OSError:
                pass

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    train_ds, val_ds, test_ds = load_image_datasets(
        dataset_root,
        image_size=image_size,
        batch_size=batch_size,
    )

    model = build_transfer_model(backbone=backbone, image_size=image_size)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )

    class_weights = {0: 1.0, 1: 1.2}

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    eval_vals = model.evaluate(test_ds, verbose=0)
    return model, {"loss": float(eval_vals[0]), "accuracy": float(eval_vals[1]), "auc": float(eval_vals[2])}


def load_pretrained_cv_model(model_path: Path) -> tf.keras.Model:
    """
    Load a saved Keras model for inference only.
    This avoids any dataset access, evaluation, or training behavior.
    """
    return tf.keras.models.load_model(model_path, compile=False)


def validate_chest_xray(uploaded_image: Image.Image) -> tuple[bool, str]:
    """
    Validate if uploaded image is likely a chest X-ray.
    Returns (is_valid, message)
    """
    try:
        rgb_image = uploaded_image.convert("RGB")
        validator = _get_xray_validator()
        predictions = validator(rgb_image, candidate_labels=XRAY_VALIDATION_LABELS, hypothesis_template="This image is {}.")

        if not predictions:
            return False, "Upload a valid chest X-ray image."

        top_prediction = predictions[0]
        top_label = str(top_prediction.get("label", "")).strip().lower()
        top_score = float(top_prediction.get("score", 0.0))
        chest_labels = {"a chest x-ray radiograph", "a chest radiograph", "a medical x-ray image"}

        chest_candidates = [item for item in predictions if str(item.get("label", "")).strip().lower() in chest_labels]
        chest_score = max((float(item.get("score", 0.0)) for item in chest_candidates), default=0.0)
        best_chest_label = next((str(item.get("label", "")).strip() for item in chest_candidates if float(item.get("score", 0.0)) == chest_score), "a chest x-ray radiograph")

        if top_label not in chest_labels or chest_score < 0.30 or chest_score < top_score - 0.05:
            return False, "Upload a valid chest X-ray image."

        return True, f"Valid chest X-ray image ({best_chest_label})"

    except Exception:
        try:
            rgb_image = uploaded_image.convert("RGB")
            img_array = np.array(rgb_image, dtype=np.float32)
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            channel_diff = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b))) / 3.0
            gray = np.mean(img_array, axis=2)
            contrast = float(np.std(gray))
            dynamic_range = float(np.percentile(gray, 98) - np.percentile(gray, 2))
            width, height = rgb_image.size
            aspect_ratio = (height / width) if width > 0 else 0.0

            if channel_diff > 28.0 and contrast < 18.0 and dynamic_range < 45.0 and not (0.65 <= aspect_ratio <= 1.8):
                return False, "Upload a valid chest X-ray image."

            return True, "Valid chest X-ray image"
        except Exception:
            return False, "Upload a valid chest X-ray image."


def preprocess_uploaded_xray(uploaded_image: Image.Image, image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    rgb_image = uploaded_image.convert("RGB")
    resized = rgb_image.resize(image_size)
    array = np.array(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_xray(model: tf.keras.Model, processed_image: np.ndarray, threshold: float = 0.70) -> Dict[str, float]:
    model_inputs = [processed_image] if len(getattr(model, "inputs", []) or []) == 1 else processed_image
    pneumonia_prob = float(model.predict(model_inputs, verbose=0)[0][0])
    normal_prob = 1.0 - pneumonia_prob

    predicted_class = "PNEUMONIA" if pneumonia_prob >= threshold else "NORMAL"
    confidence = pneumonia_prob if predicted_class == "PNEUMONIA" else normal_prob

    return {
        "predicted_class": predicted_class,
        "pneumonia_probability": pneumonia_prob,
        "normal_probability": normal_prob,
        "confidence": confidence,
    }


def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is not None and len(output_shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found for GradCAM.")


def get_conv_layer_candidates(model: tf.keras.Model) -> list[str]:
    preferred = ["top_activation", "out_relu", "Conv_1", "top_conv", "conv5_block3_out"]
    names = []

    model_layer_names = {layer.name for layer in model.layers}
    for name in preferred:
        if name in model_layer_names:
            names.append(name)

    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is not None and len(output_shape) == 4 and layer.name not in names:
            names.append(layer.name)

    if not names:
        names.append(find_last_conv_layer_name(model))

    return names


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    image_array: np.ndarray,
    pred_index: int | None = None,
) -> np.ndarray:
    model_inputs = [image_array] if len(getattr(model, "inputs", []) or []) == 1 else image_array

    for conv_layer_name in get_conv_layer_candidates(model):
        try:
            grad_model = tf.keras.models.Model(
                model.inputs,
                [model.get_layer(conv_layer_name).output, model.output],
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(model_inputs, training=False)
                if pred_index is None:
                    pred_index = 0
                class_channel = predictions[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            denom = tf.math.reduce_max(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (denom + tf.keras.backend.epsilon())
            return heatmap.numpy()
        except Exception:
            continue

    try:
        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        fallback_inputs = [image_tensor] if len(getattr(model, "inputs", []) or []) == 1 else image_tensor
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            predictions = model(fallback_inputs, training=False)
            if pred_index is None:
                pred_index = 0
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, image_tensor)
        grads = tf.abs(grads[0])
        saliency = tf.reduce_max(grads, axis=-1)
        saliency = saliency - tf.reduce_min(saliency)
        saliency = saliency / (tf.reduce_max(saliency) + tf.keras.backend.epsilon())
        return saliency.numpy()
    except Exception:
        h, w = image_array.shape[1], image_array.shape[2]
        return np.zeros((h, w), dtype=np.float32)


def overlay_heatmap_on_image(uploaded_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    rgb = np.array(uploaded_image.convert("RGB"))
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
        (rgb.shape[1], rgb.shape[0]),
        resample=Image.Resampling.BILINEAR,
    )
    heatmap_resized = np.array(heatmap_image, dtype=np.float32) / 255.0
    heatmap_color = np.stack(
        [
            np.clip(1.5 - np.abs(4.0 * heatmap_resized - offset), 0.0, 1.0)
            for offset in (3.0, 2.0, 1.0)
        ],
        axis=-1,
    )
    overlay = (1 - alpha) * (rgb.astype(np.float32) / 255.0) + alpha * heatmap_color
    return np.uint8(np.clip(overlay * 255, 0, 255))
