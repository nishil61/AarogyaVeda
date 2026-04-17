from __future__ import annotations

import base64
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "history" / "prediction_history.csv"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"


def ensure_directories() -> None:
    for directory in [
        BASE_DIR / "models",
        BASE_DIR / "history",
        BASE_DIR / "assets",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def detect_xray_dataset_root() -> Optional[Path]:
    candidates = [
        BASE_DIR / "chest_xray",
        BASE_DIR / "chest_xray" / "chest_xray",
    ]

    best_path = None
    best_count = -1

    for root in candidates:
        train_dir = root / "train"
        val_dir = root / "val"
        test_dir = root / "test"

        if not (train_dir.exists() and val_dir.exists() and test_dir.exists()):
            continue

        file_count = sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if file_count > best_count:
            best_count = file_count
            best_path = root

    return best_path


def get_image_as_base64(file_path: Path) -> str:
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def get_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "prediction_type",
                "patient_id",
                "patient_name",
                "input_summary",
                "predicted_label",
                "risk_probability",
                "confidence",
                "drive_url",
                "history_key",
            ]
        )
    return pd.read_csv(HISTORY_PATH)


def append_history(
    prediction_type: str,
    model_name: str,
    patient_name: str,
    patient_id: str,
    input_summary: str,
    predicted_label: str,
    risk_probability: float = 0.0,
    confidence: float = 0.0,
    drive_url: str = "",
    history_key: str = "",
) -> None:
    ensure_directories()

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_type": prediction_type,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "input_summary": input_summary,
        "predicted_label": predicted_label,
        "risk_probability": round(float(risk_probability), 4),
        "confidence": round(float(confidence), 4),
        "drive_url": drive_url,
        "history_key": history_key,
    }

    file_exists = HISTORY_PATH.exists()
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def clear_history() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


def update_history_with_drive_url(history_key: str, drive_url: str) -> None:
    if not HISTORY_PATH.exists():
        return

    try:
        df = pd.read_csv(HISTORY_PATH)
        if df.empty:
            return

        if "history_key" in df.columns and history_key:
            matches = df.index[df["history_key"].astype(str) == str(history_key)].tolist()
            if matches:
                df.at[matches[-1], "drive_url"] = drive_url
            else:
                df.at[len(df) - 1, "drive_url"] = drive_url
        else:
            df.at[len(df) - 1, "drive_url"] = drive_url

        df.to_csv(HISTORY_PATH, index=False)
    except Exception as e:
        print(f"Error updating history with drive URL: {e}")
