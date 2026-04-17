from __future__ import annotations

import os

# Set environment variables BEFORE any imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

import base64
import io
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import InferenceClient, get_token
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib import colors
from dotenv import load_dotenv
from PIL import Image
from app_utils import get_local_now
from functools import lru_cache

try:
    import streamlit as st
except Exception:
    st = None

load_dotenv()


def _get_config_value(key: str, default=None):
    value = os.getenv(key)
    if value not in (None, ""):
        return value

    if st is not None:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass

    return default


HF_TOKEN = _get_config_value("HF_TOKEN") or _get_config_value("HUGGINGFACE_API_KEY")
TEXT_MODEL_ID = _get_config_value("HF_TEXT_MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct")
OPENROUTER_API_KEY = (
    _get_config_value("OPENROUTER_API_KEY", "")
    or _get_config_value("OPEN_ROUTER_API_KEY", "")
)
OPENROUTER_BASE_URL = _get_config_value("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL_ID = _get_config_value("OPENROUTER_MODEL_ID", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_MODEL_IDS = [
    model_id.strip()
    for model_id in str(
        _get_config_value(
            "OPENROUTER_MODEL_IDS",
            "meta-llama/llama-3.3-70b-instruct:free,meta-llama/llama-3.1-70b-instruct:free,meta-llama/llama-3.1-8b-instruct:free",
        )
    ).split(",")
    if model_id.strip()
]
OPENROUTER_HTTP_REFERER = _get_config_value("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_X_TITLE = _get_config_value("OPENROUTER_X_TITLE", "AarogyaVeda")
OPENROUTER_MAX_TOKENS = int(str(_get_config_value("OPENROUTER_MAX_TOKENS", "1800") or "1800").strip())
IMAGE_CAPTION_MODEL_ID = _get_config_value("HF_IMAGE_CAPTION_MODEL_ID", "Salesforce/blip-image-captioning-base")
IMAGE_CLASSIFICATION_MODEL_ID = _get_config_value("HF_IMAGE_CLASSIFICATION_MODEL_ID", "google/vit-base-patch16-224")
SIGN_CANDIDATES = ["AarogyaVeda Sign.png"]
WATERMARK_SOURCE_CANDIDATES = [
    "Logo.png",
    "AarogyaVeda logo landscape.png",
    "AarogyaVeda_logo_landscape.png",
    "AarogyaVeda logo.png",
]


def _resolve_sign_path() -> str | None:
    base = Path(__file__).resolve().parent
    for filename in SIGN_CANDIDATES:
        candidate = base / filename
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_watermark_path() -> str | None:
                                                                                     
    generated = _ensure_watermark_assets()
    if generated:
        return generated[1]

    return _resolve_sign_path()


def _ensure_watermark_assets() -> tuple[str, str] | None:
    base = Path(__file__).resolve().parent
    assets_dir = base / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_png = assets_dir / "aarogyaveda_watermark.png"

    source = None
    for filename in WATERMARK_SOURCE_CANDIDATES:
        candidate = base / filename
        if candidate.exists():
            source = candidate
            break
    if source is None:
        sign_path = _resolve_sign_path()
        source = Path(sign_path) if sign_path else None
    if source is None or not source.exists():
        return None

    img = Image.open(source).convert("RGBA")
    pixels = []
    blend = 0.12
    for r, g, b, a in img.getdata():
        if a < 20:
            pixels.append((r, g, b, 0))
            continue
        if (r < 24 and g < 24 and b < 24) or (r > 246 and g > 246 and b > 246):
            pixels.append((r, g, b, 0))
        else:
                                                                                             
            fr = int(255 - (255 - r) * blend)
            fg = int(255 - (255 - g) * blend)
            fb = int(255 - (255 - b) * blend)
            pixels.append((fr, fg, fb, 255))
    img.putdata(pixels)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    canvas_w, canvas_h = 3600, 1200
    canvas_img = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    scale = min((canvas_w * 0.88) / max(img.width, 1), (canvas_h * 0.72) / max(img.height, 1))
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    canvas_img.alpha_composite(img, (x, y))
    canvas_img.save(out_png)

                                                                     
    return "", str(out_png)


def _format_time_12h(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def _image_to_data_url(image: Image.Image, max_side: int = 768) -> str:
    rgb = image.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, float(max_side) / max(max(w, 1), max(h, 1)))
    if scale < 1.0:
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _normalize_llm_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", cleaned)
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned, flags=re.MULTILINE)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    filtered_lines: list[str] = []
    for line in lines:
        lowered = line.strip().lower().replace("&", "and")
        if lowered in {
            "<b>clinical recommendations and patient precautions</b>",
            "clinical recommendations and patient precautions",
            "clinical recommendations & patient precautions",
        }:
            continue
        filtered_lines.append(line)
    cleaned = "\n".join(filtered_lines)
    cleaned = re.sub(r"^\s*\*\s+", "• ", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def _extract_section(text: str, section_name: str, next_sections: list[str]) -> str:
    """
    Robustly extract a section from LLM output.
    Handles various formatting: "SECTION:", "SECTION :", with/without colons, etc.
    """
    if not text:
        return ""
    
                                               
    section_start = None
    section_pattern = rf"(?:^|\n)\s*{re.escape(section_name)}\s*:?\s*(?:\n|$)"
    
    for match in re.finditer(section_pattern, text, flags=re.IGNORECASE | re.MULTILINE):
        section_start = match.end()
        break
    
    if section_start is None:
        return ""
    
                                    
    section_end = len(text)
    for next_sec in next_sections:
        next_pattern = rf"(?:^|\n)\s*{re.escape(next_sec)}\s*:?\s*(?:\n|$)"
        for match in re.finditer(next_pattern, text[section_start:], flags=re.IGNORECASE | re.MULTILINE):
            end_pos = section_start + match.start()
            if end_pos < section_end:
                section_end = end_pos
            break
    
    content = text[section_start:section_end].strip()
                                                      
    content = re.sub(rf"^\s*(?:{section_name}|IMPRESSION|PRECAUTIONS|FINDINGS)\s*:?\s*", "", content, flags=re.IGNORECASE)
    return content.strip()


def _remove_numeric_factors(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*%\b", "", text)
    cleaned = re.sub(r"\b(model\s+confidence|confidence|probability|score|activation|ratio)\b[^\n.]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _enforce_numbered_precautions(precautions: str, min_count: int = 5) -> str:
    """
    Strictly enforce precautions into numbered point-wise format starting at 1.
    Respects natural sentence boundaries - does NOT break compound statements.
    """
    if not precautions:
        return ""
    
    precautions = precautions.strip()
    
    points = []
    
    # First, try to extract already-numbered points
    numbered_pattern = r'^\s*\d+\.\s+'
    current_point = ""
    for line in precautions.splitlines():
        line = line.strip()
        if not line:
            continue
        
        # If line starts with a number, it's a new point
        if re.match(numbered_pattern, line):
            if current_point:
                points.append(current_point)
            cleaned = re.sub(r'^[\s\d\.\)\:\-•*]+\s*', '', line)
            current_point = cleaned
        else:
            # Add to current point (allows multi-sentence points)
            if current_point:
                current_point += " " + line
            else:
                current_point = line
    
    # Don't forget the last point
    if current_point:
        points.append(current_point)
    
    # If still no points or very few, split by sentences to create points
    if len(points) < 2:
        text = precautions.strip()
        text = re.sub(r'^[\s\d\.\)\:\-•*]+\s*', '', text)
        
        # Split by periods but group related sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into points (allow 2-3 sentences per point)
        if sentences:
            current_group = sentences[0]
            for sent in sentences[1:]:
                if len(current_group.split()) < 60:  # Allow up to 60 words per point
                    current_group += " " + sent
                else:
                    if current_group:
                        points.append(current_group)
                    current_group = sent
            if current_group:
                points.append(current_group)
    
    # Check if existing points are image-specific
    image_keywords = ['consolidation', 'pneumonia', 'infiltrate', 'opacity', 'lobe', 'zone', 'region', 'heatmap', 'affected', 'involvement', 'imaging', 'radiographic', 'finding', 'bilateral', 'unilateral', 'hemithorax', 'pleural', 'cavitation', 'density', 'attenuation', 'infiltration']
    existing_text = " ".join(points).lower()
    has_image_specific = any(keyword in existing_text for keyword in image_keywords)
    
    # Only add defaults if existing points are generic
    if len(points) < min_count and not has_image_specific:
        default_additions = [
            "Follow physician guidance carefully and attend all scheduled follow-up appointments.",
            "Monitor symptoms daily and report any worsening signs immediately to your healthcare provider.",
            "Maintain adequate hydration and ensure sufficient rest to support recovery.",
            "Take all prescribed medications exactly as directed, completing the full course.",
            "Avoid exposure to smoke, dust, and other respiratory irritants.",
            "Seek urgent medical attention for severe breathing difficulty, chest pain, or signs of deterioration.",
        ]
        for addition in default_additions:
            if len(points) >= min_count:
                break
            if addition.lower()[:30] not in existing_text:
                points.append(addition)
    
    result_lines = []
    for i, point in enumerate(points[:15], 1):
        content = point.strip()
        
        # Remove any leading numbers/bullets
        content = re.sub(r'^[\d\.\)\:\-•*\s]+', '', content).strip()
        
        # Ensure ends with punctuation
        if content and not content.endswith(('.', '!', '?')):
            content += '.'
        
        if content:
            result_lines.append(f"{i}. {content}")
    
    return "\n".join(result_lines)


def _rewrite_report_sections(
    client: InferenceClient,
    findings: str,
    impression: str,
    precautions: str,
    severity: str,
    case_context: str,
) -> tuple[str, str, str]:
    rewrite_system = (
        "You are a clinical report quality editor. Rewrite content to strictly follow style requirements. "
        "Use the term 'patient' only. Avoid technical numeric factors, percentages, and model metrics."
    )

    rewrite_prompt = (
        "Rewrite the three sections below with the following strict rules:\n"
        "1) Use only the common word 'patient' (no names).\n"
        "2) Keep content detailed and clinically useful, not short summaries.\n"
        "3) Keep findings and impression narrative and case-specific.\n"
        "4) PRECAUTIONS must be point-wise and numbered.\n"
        "5) Do not include percentages, scores, or model technical factors.\n"
        f"6) Severity context: {severity}.\n"
        f"7) Image context: {case_context if case_context else 'No additional context available.'}\n\n"
        "Output exactly in this format:\n"
        "FINDINGS:\n...\n\nIMPRESSION:\n...\n\nPRECAUTIONS:\n1. ...\n2. ...\n"
        "Current sections to rewrite:\n\n"
        f"FINDINGS:\n{findings}\n\n"
        f"IMPRESSION:\n{impression}\n\n"
        f"PRECAUTIONS:\n{precautions}\n"
    )

    resp = client.chat_completion(
        messages=[
            {"role": "system", "content": rewrite_system},
            {"role": "user", "content": rewrite_prompt},
        ],
        max_tokens=1300,
        temperature=0.45,
    )
    rewritten = (
        resp.choices[0].message.content.strip()
        if hasattr(resp, "choices") and resp.choices
        else ""
    )

    f2 = _extract_section(rewritten, "FINDINGS", ["IMPRESSION", "PRECAUTIONS"])
    i2 = _extract_section(rewritten, "IMPRESSION", ["PRECAUTIONS"])
    p2 = _extract_section(rewritten, "PRECAUTIONS", ["END"])
    return f2 or findings, i2 or impression, p2 or precautions


def get_inference_client(model_id: str | None = None) -> InferenceClient:
    token = HF_TOKEN or get_token()
    if not token:
        raise RuntimeError("Set HF_TOKEN in .env or run hf auth login first.")
    return InferenceClient(model=model_id or TEXT_MODEL_ID, token=token)


@lru_cache(maxsize=1)
def _get_openrouter_client():
    api_key = str(OPENROUTER_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required for OpenRouter fallback.") from exc

    return OpenAI(
        api_key=api_key,
        base_url=(OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1").rstrip("/"),
    )


def _chat_completion_with_fallback(
    primary_client: InferenceClient,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    openrouter_messages: list[dict[str, str]] | None = None,
) -> tuple[Any, str]:
    try:
        response = primary_client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response, TEXT_MODEL_ID
    except Exception as hf_exc:
        if not str(OPENROUTER_API_KEY or "").strip():
            raise hf_exc

        openrouter_client = _get_openrouter_client()
        extra_headers: dict[str, str] = {}
        if OPENROUTER_HTTP_REFERER:
            extra_headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        if OPENROUTER_X_TITLE:
            extra_headers["X-Title"] = OPENROUTER_X_TITLE

        request_messages = openrouter_messages or messages
        openrouter_errors: list[str] = []
        model_ids = OPENROUTER_MODEL_IDS or [OPENROUTER_MODEL_ID]
        last_openrouter_exc: Exception | None = None

        for model_id in model_ids:
            try:
                response = openrouter_client.chat.completions.create(
                    model=model_id,
                    messages=request_messages,
                    max_tokens=min(max_tokens, max(256, OPENROUTER_MAX_TOKENS)),
                    temperature=temperature,
                    extra_headers=extra_headers or None,
                )
                return response, f"openrouter:{model_id}"
            except Exception as openrouter_exc:
                last_openrouter_exc = openrouter_exc
                openrouter_errors.append(f"{model_id}: {openrouter_exc}")

        raise RuntimeError(
            f"HF generation failed: {hf_exc}. OpenRouter fallback failed: {' | '.join(openrouter_errors)}"
        ) from last_openrouter_exc


def _extract_message_text(response: Any) -> str:
    if not hasattr(response, "choices") or not response.choices:
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if txt:
                    parts.append(str(txt))
            else:
                txt = getattr(item, "text", None)
                if txt:
                    parts.append(str(txt))
        return "\n".join(parts).strip()
    return str(content).strip()


def _extract_caption_text(caption_response: Any) -> str:
    if isinstance(caption_response, str):
        return caption_response.strip()
    if isinstance(caption_response, dict):
        for key in ("generated_text", "text", "caption"):
            value = caption_response.get(key)
            if value:
                return str(value).strip()
    if isinstance(caption_response, list) and caption_response:
        first = caption_response[0]
        if isinstance(first, dict):
            for key in ("generated_text", "text", "caption"):
                value = first.get(key)
                if value:
                    return str(value).strip()
        if isinstance(first, str):
            return first.strip()
    return ""


@lru_cache(maxsize=2)
def _get_local_caption_pipeline():
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("transformers is required for local caption fallback") from exc
    return pipeline("image-to-text", model=IMAGE_CAPTION_MODEL_ID)


@lru_cache(maxsize=2)
def _get_local_classification_pipeline():
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("transformers is required for local image-classification fallback") from exc
    return pipeline("image-classification", model=IMAGE_CLASSIFICATION_MODEL_ID)


def _analyze_gradcam_heatmap(gradcam_image: Image.Image) -> str:
    """Analyze Grad-CAM heatmap to describe model attention regions."""
    try:
        import numpy as np
        
        # Convert to RGB if needed, then to array
        heatmap_array = np.array(gradcam_image.convert("RGB"))
        
        # Convert to grayscale for intensity analysis
        gray = np.mean(heatmap_array, axis=2)
        
        # Find regions above a threshold (bright areas = high attention)
        threshold = np.percentile(gray, 60)
        high_attention = gray > threshold
        
        h, w = high_attention.shape
        h_mid = h // 2
        w_mid = w // 2
        
        # Divide into quadrants: upper-left, upper-right, lower-left, lower-right
        ul = high_attention[:h_mid, :w_mid].sum()
        ur = high_attention[:h_mid, w_mid:].sum()
        ll = high_attention[h_mid:, :w_mid].sum()
        lr = high_attention[h_mid:, w_mid:].sum()
        
        regions = []
        if ul > (h_mid * w_mid * 0.15):  # 15% threshold
            regions.append("left upper region")
        if ur > (h_mid * w_mid * 0.15):
            regions.append("right upper region")
        if ll > (h_mid * w_mid * 0.15):
            regions.append("left lower region")
        if lr > (h_mid * w_mid * 0.15):
            regions.append("right lower region")
        
        if not regions:
            return ""
        
        region_str = ", ".join(regions)
        return f"Model attention concentrated in: {region_str}."
    except Exception:
        return ""


def _classification_to_context(classification_result: Any) -> str:
    if not isinstance(classification_result, list) or not classification_result:
        return ""

    medical_keywords = {
        "x-ray", "xray", "radiograph", "chest", "lung", "thorax", "pneumonia",
        "opacity", "infiltrate", "consolidation", "pleural", "respiratory", "medical",
    }

    relevant_labels: list[str] = []
    for item in classification_result[:5]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        if not label:
            continue
        if any(keyword in label for keyword in medical_keywords):
            relevant_labels.append(label)

    if not relevant_labels:
        return ""

    merged = ", ".join(dict.fromkeys(relevant_labels))
    return f"Image classifier detected thoracic/radiographic cues: {merged}."


def _generate_image_caption(uploaded_image: Image.Image) -> tuple[str, str, str]:
    """Generate caption using local models only (no hosted inference)."""
    try:
        local_captioner = _get_local_caption_pipeline()
        result = local_captioner(uploaded_image.convert("RGB"))
        caption_text = _extract_caption_text(result)
        if caption_text:
            return caption_text, f"local-caption:{IMAGE_CAPTION_MODEL_ID}", ""
    except Exception:
        pass

    try:
        local_classifier = _get_local_classification_pipeline()
        cls_result = local_classifier(uploaded_image.convert("RGB"))
        cls_context = _classification_to_context(cls_result)
        if cls_context:
            return cls_context, f"local-classification:{IMAGE_CLASSIFICATION_MODEL_ID}", ""
    except Exception:
        pass

    return "", "none", ""


def _draw_pdf_header_footer(canvas, doc) -> None:
    canvas.saveState()
    page_width, page_height = letter
    canvas.setFillColor(colors.HexColor("#0B3A5B"))
    canvas.rect(0, page_height - 0.65 * inch, page_width, 0.65 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(doc.leftMargin, page_height - 0.38 * inch, "AarogyaVeda Medical Center")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(page_width - doc.rightMargin, page_height - 0.38 * inch, "Clinical Imaging Report")

    canvas.setStrokeColor(colors.HexColor("#C0C0C0"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 0.55 * inch, page_width - doc.rightMargin, 0.55 * inch)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.setFont("Helvetica", 8)
    now = get_local_now()
    canvas.drawString(doc.leftMargin, 0.35 * inch, f"Generated on {now.strftime('%B %d, %Y')} {_format_time_12h(now)}")
    canvas.drawRightString(page_width - doc.rightMargin, 0.35 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def _draw_pdf_watermark(canvas, doc) -> None:
    watermark_path = _resolve_watermark_path()
    if not watermark_path:
        return

    canvas.saveState()
    page_width, page_height = letter
    try:
        canvas.translate(page_width / 2, page_height / 2)
        canvas.rotate(28)

        target_w = 7.8 * inch
        target_h = 2.7 * inch
        canvas.drawImage(
            watermark_path,
            -target_w / 2,
            -target_h / 2,
            width=target_w,
            height=target_h,
            preserveAspectRatio=True,
            mask="auto",
        )
    except Exception:
        pass
    canvas.restoreState()


def _draw_pdf_page(canvas, doc) -> None:
    _draw_pdf_watermark(canvas, doc)
    _draw_pdf_header_footer(canvas, doc)


def generate_medical_report_content(
    prediction_label: str,
    confidence: float,
    patient_name: str = "Patient",
    patient_id: str = "N/A",
    case_context: str = "",
    uploaded_image: Image.Image | None = None,
    gradcam_image: Image.Image | None = None,
) -> Dict[str, str]:
    """
    Generate detailed clinical report content using the LLM based on X-ray prediction.
    prediction_label: "PNEUMONIA" or "NORMAL"
    confidence: float between 0 and 1
    """
    
    client = get_inference_client(TEXT_MODEL_ID)
    generation_mode = "image_assisted_text" if (uploaded_image is not None or gradcam_image is not None) else "text"
    model_used = TEXT_MODEL_ID
    caption_source = ""
    is_severe = prediction_label.upper() == "PNEUMONIA"
    severity = "High" if is_severe else "Low"
    min_precaution_points = 7 if is_severe else 5
    context_text = case_context if case_context else "Comprehensive chest X-ray review without additional contextual annotations."
    openrouter_system_prompt = (
        "You are an expert radiologist. Write a clinical chest X-ray report with these sections only: "
        "FINDINGS, IMPRESSION, PRECAUTIONS. Use only the word patient. Do not mention confidence, scores, or model terms."
    )

    system_prompt = (
        "You are an expert radiologist writing detailed chest X-ray reports. "
        "Do not use canned phrases or repetitive stock wording. "
        "Every report must be case-specific and textually distinct. "
        "Use only the generic term 'patient'. "
        "Avoid AI/model metrics, percentages, and confidence language. "
        "Only mention clinically plausible chest imaging findings. "
        "Never introduce unrelated objects, metaphors, or speculative non-medical narratives."
    )

    def word_count(txt: str) -> int:
        return len(re.findall(r"[A-Za-z0-9]+", txt or ""))

    def _sanitize_sections(findings: str, impression: str, precautions: str) -> tuple[str, str, str]:
        if patient_name and patient_name.strip() and patient_name.strip().lower() != "patient":
            pattern = re.escape(patient_name.strip())
            findings = re.sub(pattern, "patient", findings, flags=re.IGNORECASE)
            impression = re.sub(pattern, "patient", impression, flags=re.IGNORECASE)
            precautions = re.sub(pattern, "patient", precautions, flags=re.IGNORECASE)

        findings = re.sub(r"\b(the patient named|patient name)\b.*", "patient", findings, flags=re.IGNORECASE)
        impression = re.sub(r"\b(the patient named|patient name)\b.*", "patient", impression, flags=re.IGNORECASE)
        precautions = re.sub(r"\b(the patient named|patient name)\b.*", "patient", precautions, flags=re.IGNORECASE)

        findings = _remove_numeric_factors(findings)
        impression = _remove_numeric_factors(impression)
        precautions = _remove_numeric_factors(precautions)
        precautions = _enforce_numbered_precautions(precautions, min_precaution_points)
        return findings.strip(), impression.strip(), precautions.strip()

    best_sections = ("", "", "")
    best_score = -1
    last_error = ""

    for attempt in range(3):
        try:
            temperature = 0.85 + (attempt * 0.12)

            combined_prompt = (
                "Generate an EXTREMELY DETAILED, COMPREHENSIVE, and LENGTHY chest X-ray report for this single case.\n"
                "CRITICAL: Reports must be very detailed with NO abbreviations, NO brevity. Every section must be LONG, THOROUGH, and FULL OF CLINICAL DETAIL.\n\n"
                f"IMAGE CONTEXT:\n{context_text}\n\n"
                f"Assessment label: {prediction_label}\n"
                f"Severity context: {severity}\n"
                "\n"
                "LENGTH REQUIREMENTS (STRICT):\n"
                "- FINDINGS section: MINIMUM 250+ words. Describe every region, every abnormality, texture, density, distribution, bilateral patterns, specific locations. NO short descriptions.\n"
                "- IMPRESSION section: MINIMUM 180+ words. Provide comprehensive clinical correlation, disease severity assessment, differential considerations, prognostic implications.\n"
                "- PRECAUTIONS section: EXACTLY 7 NUMBERED POINTS (each 1 sentence, but with maximum allowed detail per sentence).\n\n"
                "FORMAT (exactly these section headers):\n"
                "FINDINGS:\n"
                "[VERY DETAILED, LONG narrative text - minimum 250 words, describe every finding in detail]\n\n"
                "IMPRESSION:\n"
                "[VERY DETAILED, LONG narrative text - minimum 180 words, comprehensive clinical analysis]\n\n"
                "PRECAUTIONS:\n"
                "1. [Detailed sentence with image reference]\n"
                "2. [Detailed sentence with image reference]\n"
                "3. [Detailed sentence with image reference]\n"
                "4. [Detailed sentence with image reference]\n"
                "5. [Detailed sentence with image reference]\n"
                "6. [Detailed sentence with image reference]\n"
                "7. [Detailed sentence with image reference]\n\n"
                "Rules:\n"
                "- FINDINGS: Be EXTREMELY THOROUGH. Describe opacity characteristics, distribution across lobes, involved regions (upper lobe, lower lobe, lingula, specific zones), texture, density variations, bronchogram patterns, pleural involvement, cardiac silhouette, mediastinal structures, diaphragm, costophrenic angles. Use detailed anatomical descriptions.\n"
                "- IMPRESSION: COMPREHENSIVE analysis. Include differential diagnosis discussion, severity assessment, distribution pattern analysis, clinical correlation, complications risk, prognosis implications. NO brief summaries.\n"
                "- PRECAUTIONS: Output EXACTLY 7 NUMBERED POINTS, EACH ON A SEPARATE LINE. Each point should be 2-3 DETAILED SENTENCES addressing a specific clinical concern related to the imaging findings.\n"
                "- CRITICAL: Each precaution point should be COMPREHENSIVE (2-3 sentences, not a one-liner). Cover the specific imaging finding, the clinical implication, and the recommended action/monitoring.\n"
                "- CRITICAL: EVERY precaution point MUST mention a specific imaging finding from this case (e.g., 'consolidation in mid-central zone', 'bilateral involvement', 'right hemithorax', specific lobes, specific regions, severity level, distribution patterns).\n"
                "- PRECAUTION POINT STRUCTURE: [Finding reference] + [Clinical implication/risk] + [Specific action/monitoring recommendation]\n"
                "- Examples of CORRECT format (2-3 sentences each):\n"
                "  1. The mid-central consolidation with high-density characteristics warrants close monitoring for signs of disease progression or potential cavitation development. Serial imaging should be considered to assess treatment response. Respiratory status and oxygen requirements must be closely tracked.\n"
                "  2. The bilateral upper lobe involvement with significant consolidation requires aggressive antibiotic therapy initiation and close clinical monitoring. Laboratory investigations including blood cultures and sputum analysis should be performed. Close observation for signs of clinical deterioration including worsening hypoxemia is essential.\n"
                "  3. The right hemithorax dominance with greater opacification necessitates specific surveillance for evolving oxygen requirements and potential respiratory complications. Pulse oximetry monitoring should be continuous. The patient should be assessed for signs of respiratory distress or hypoxemia requiring intervention.\n"
                "  4. Given the multifocal pattern and overall extent of infiltration, follow-up chest imaging in 7-10 days is essential to assess treatment response. Clinical correlation with symptoms should guide imaging decisions. Advanced imaging may be needed if clinical response is suboptimal.\n"
                "  5. The distribution pattern of consolidation suggesting community-acquired pneumonia warrants thorough clinical correlation with patient symptoms and physical examination findings. Fever curves, sputum production, and respiratory symptoms should be tracked. Negative workup for atypical organisms may be needed based on clinical presentation.\n"
                "  6. Vigilant observation for potential complications including pleural effusion or pneumothorax development is warranted given disease severity. Chest auscultation should be performed regularly. Any new pleural findings should prompt repeat imaging and possible intervention.\n"
                "  7. Given the multifocal and high-severity pneumonic process, consider advanced imaging such as CT if clinical deterioration occurs despite appropriate antimicrobial therapy. CT would help evaluate for complications and guide further management. Close follow-up and reassessment parameters should be established.\n"
                "- Examples of WRONG format (DO NOT OUTPUT):\n"
                "  BAD: '1. Monitor respiratory status.' (too generic, no imaging reference, too short)\n"
                "  BAD: '1. The patient must exercise caution due to pneumonic consolidation....' (vague, too broad)\n"
                "- Precautions must be DISTINCT - no point should repeat findings/recommendations from other points.\n"
                "- All sections must demonstrate deep understanding of this patient's specific radiographic presentation.\n"
                "- Keep clinically meaningful and evidence-based.\n"
                "- Ignore any non-medical object labels if they appear in image-derived metadata.\n"
                "- Do not include metaphors or references unrelated to pulmonary/chest disease.\n"
                "- NEVER abbreviate or shorten sections. Generate FULL LENGTH across all areas."
            )

            # Use image-assisted text model directly (no vision model fallback)
            if uploaded_image is not None or gradcam_image is not None:
                try:
                    caption, detected_caption_source, _ = _generate_image_caption(uploaded_image) if uploaded_image is not None else ("", "none", "")
                    caption_source = detected_caption_source
                except Exception:
                    caption = ""
                    caption_source = "none"
                
                gradcam_analysis = (
                    _analyze_gradcam_heatmap(gradcam_image)
                    if gradcam_image is not None
                    else ""
                )
                gradcam_note = ""
                if gradcam_analysis:
                    gradcam_note = f"Grad-CAM spotlight: {gradcam_analysis} Use this localization evidence to focus the report on affected regions.\n"
                elif gradcam_image is not None:
                    gradcam_note = "Grad-CAM heatmap available: Use the model-identified focus regions to prioritize clinical analysis.\n"
                caption_note = f"{caption}\n" if caption else ""
                compact_openrouter_prompt = (
                    "Generate a detailed chest X-ray report for one patient. "
                    "Return exactly these sections with headings: FINDINGS, IMPRESSION, PRECAUTIONS. "
                    "Keep findings and impression case-specific and clinically plausible. "
                    "PRECAUTIONS must be exactly 7 numbered points, each tied to an imaging finding.\n\n"
                    f"Assessment label: {prediction_label}\n"
                    f"Severity context: {severity}\n"
                    f"Image context: {context_text}\n\n"
                    "IMAGE-DERIVED CONTEXT:\n"
                    f"{caption_note}"
                    f"{gradcam_note}"
                    "Ignore any non-medical labels. Focus strictly on chest radiographic findings."
                )
                image_assisted_prompt = (
                    f"{combined_prompt}\n\n"
                    "IMAGE-DERIVED CONTEXT:\n"
                    f"{caption_note}"
                    f"{gradcam_note}"
                    "If any image-derived text appears non-medical or irrelevant, ignore it and focus strictly on chest radiographic disease features."
                )
                report_response, provider_model_used = _chat_completion_with_fallback(
                    primary_client=client,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": image_assisted_prompt},
                    ],
                    max_tokens=6000,
                    temperature=temperature,
                    openrouter_messages=[
                        {"role": "system", "content": openrouter_system_prompt},
                        {"role": "user", "content": compact_openrouter_prompt},
                    ],
                )
                model_used = provider_model_used
            else:
                compact_openrouter_prompt = (
                    "Generate a detailed chest X-ray report for one patient. "
                    "Return exactly these sections with headings: FINDINGS, IMPRESSION, PRECAUTIONS. "
                    "PRECAUTIONS must be exactly 7 numbered points.\n\n"
                    f"Assessment label: {prediction_label}\n"
                    f"Severity context: {severity}\n"
                    f"Image context: {context_text}\n"
                )
                report_response, provider_model_used = _chat_completion_with_fallback(
                    primary_client=client,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": combined_prompt},
                    ],
                    max_tokens=6000,
                    temperature=temperature,
                    openrouter_messages=[
                        {"role": "system", "content": openrouter_system_prompt},
                        {"role": "user", "content": compact_openrouter_prompt},
                    ],
                )
                model_used = provider_model_used
            full_report = _extract_message_text(report_response)

            findings = _extract_section(full_report, "FINDINGS", ["IMPRESSION", "PRECAUTIONS"])
            impression = _extract_section(full_report, "IMPRESSION", ["PRECAUTIONS", "RECOMMENDATIONS"])
            precautions = _extract_section(full_report, "PRECAUTIONS", ["RECOMMENDATIONS", "END"])

            missing_sections = []
            if word_count(findings) < 200:
                missing_sections.append("FINDINGS")
            if word_count(impression) < 140:
                missing_sections.append("IMPRESSION")
            if word_count(precautions) < 80:
                missing_sections.append("PRECAUTIONS")

            if missing_sections:
                refill_prompt = (
                    "Some sections were too short or missing. Regenerate ONLY these sections with deeper, case-specific detail: "
                    f"{', '.join(missing_sections)}.\n\n"
                    f"IMAGE CONTEXT:\n{context_text}\n\n"
                    "Return only requested sections with their headers."
                )
                refill_response, provider_model_used = _chat_completion_with_fallback(
                    primary_client=client,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": refill_prompt},
                    ],
                    max_tokens=3500,
                    temperature=min(1.2, temperature + 0.1),
                    openrouter_messages=[
                        {"role": "system", "content": openrouter_system_prompt},
                        {"role": "user", "content": refill_prompt},
                    ],
                )
                model_used = provider_model_used
                refill_text = (
                    refill_response.choices[0].message.content.strip()
                    if hasattr(refill_response, "choices") and refill_response.choices
                    else ""
                )

                if "FINDINGS" in missing_sections:
                    findings_refill = _extract_section(refill_text, "FINDINGS", ["IMPRESSION", "PRECAUTIONS"])
                    findings = findings_refill or findings
                if "IMPRESSION" in missing_sections:
                    impression_refill = _extract_section(refill_text, "IMPRESSION", ["PRECAUTIONS", "RECOMMENDATIONS"])
                    impression = impression_refill or impression
                if "PRECAUTIONS" in missing_sections:
                    precautions_refill = _extract_section(refill_text, "PRECAUTIONS", ["RECOMMENDATIONS", "END"])
                    precautions = precautions_refill or precautions

            findings, impression, precautions = _sanitize_sections(findings, impression, precautions)

            score = word_count(findings) + word_count(impression) + word_count(precautions)
            if score > best_score:
                best_sections = (findings, impression, precautions)
                best_score = score

            if (
                word_count(findings) >= 300
                and word_count(impression) >= 200
                and word_count(precautions) >= 120
            ):
                break

        except Exception as exc:
            last_error = str(exc)

    findings, impression, precautions = best_sections
    if not findings or not impression or not precautions:
        if last_error:
            compact = re.sub(r"\s+", " ", str(last_error)).strip()
            error_msg = compact[:520]
        else:
            error_msg = "generation unavailable"
        return {
            "findings": f"Report generation unavailable: {error_msg}",
            "impression": "Unable to generate impression at this time.",
            "precautions": "1. Please retry report generation.",
            "generation_mode": "error",
            "model_used": model_used,
            "caption_source": caption_source,
        }

    return {
        "findings": findings,
        "impression": impression,
        "precautions": precautions,
        "generation_mode": generation_mode,
        "model_used": model_used,
        "caption_source": caption_source,
        "images_analyzed": f"uploaded_xray + gradcam_heatmap" if gradcam_image is not None else "uploaded_xray_only" if uploaded_image is not None else "no_images",
    }


def create_hospital_report_pdf(
    prediction_label: str,
    confidence: float,
    report_content: Dict[str, str],
    patient_name: str = "Patient",
    patient_id: str = "N/A",
    doctor_name: str = "Dr. Medical AI",
    hospital_name: str = "AarogyaVeda Medical Center",
) -> bytes:
    """
    Create a professional hospital-style PDF report.
    Returns PDF as bytes that can be downloaded.
    """
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=1.05 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#003366"),
        spaceAfter=6,
        alignment=1,
    )

    section_box_style = ParagraphStyle(
        "SectionBox",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.white,
        spaceAfter=6,
        spaceBefore=12,
        backColor=colors.HexColor("#0B3A5B"),
        borderPadding=6,
    )

    stamp_style = ParagraphStyle(
        "StampStyle",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#B00020"),
        alignment=1,
        borderColor=colors.HexColor("#B00020"),
        borderWidth=1.4,
        borderPadding=8,
        leading=16,
        spaceAfter=10,
        backColor=colors.HexColor("#FFF1F2"),
    )
    
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#003366"),
        spaceAfter=6,
        spaceBefore=12,
        borderPadding=5,
        backColor=colors.HexColor("#E8F0F7"),
    )
    
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=10,
        alignment=4,
        spaceAfter=6,
        leading=14,
    )
    
    story = []
    
    report_id = f"AV-{get_local_now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    report_dt = get_local_now()

    story.append(Paragraph("RADIOLOGY REPORT", title_style))
    story.append(Paragraph(hospital_name, styles["Normal"]))
    story.append(Paragraph("Department of Radiology and Clinical Imaging", styles["Normal"]))
    story.append(Spacer(1, 0.15 * inch))
    
    header_data = [
        ["Report ID:", report_id, "Report Type:", "Chest X-Ray"],
        ["Patient Name:", patient_name, "Patient ID:", patient_id],
        ["Report Date:", report_dt.strftime("%B %d, %Y"), "Report Time:", _format_time_12h(report_dt)],
        ["Modality:", "Chest X-Ray", "Radiologist:", doctor_name],
    ]
    
    header_table = Table(header_data, colWidths=[1.5 * inch, 2 * inch, 1.5 * inch, 2 * inch])
    header_table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F0F0")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("FINAL REPORT", stamp_style))
    
    pred_color = colors.HexColor("#CC0000") if prediction_label == "PNEUMONIA" else colors.HexColor("#006600")
    pred_text = f"<b>Assessment: <font color={pred_color.hexval()}>{prediction_label}</font></b>"
    
    story.append(Paragraph(pred_text, body_style))
    story.append(Spacer(1, 0.15 * inch))
    
    story.append(Paragraph("CLINICAL FINDINGS", heading_style))
    findings_text = _normalize_llm_text(report_content.get("findings", "No findings available."))
    story.append(Paragraph(findings_text or "No findings available.", body_style))
    story.append(Spacer(1, 0.1 * inch))
    
    story.append(Paragraph("RADIOLOGICAL IMPRESSION", section_box_style))
    impression_text = _normalize_llm_text(report_content.get("impression", "No impression available."))
    story.append(Paragraph(impression_text or "No impression available.", body_style))
    story.append(Spacer(1, 0.1 * inch))
    
    story.append(Paragraph("CLINICAL RECOMMENDATIONS & PATIENT PRECAUTIONS", heading_style))
    precautions_text = _normalize_llm_text(report_content.get("precautions", "Consult with your physician."))
    for line in precautions_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), body_style))
    
    story.append(Spacer(1, 0.2 * inch))

    story.append(Spacer(1, 0.25 * inch))
    signature_logo = _resolve_sign_path()
    if signature_logo:
        try:
            logo_image = RLImage(signature_logo, width=2.8 * inch, height=0.75 * inch)
            logo_image.hAlign = "LEFT"
            story.append(logo_image)
            story.append(Spacer(1, 0.03 * inch))
        except Exception:
            pass
    story.append(Paragraph("______________________________", styles["Normal"]))
    story.append(Paragraph(f"Authorized Signature: {doctor_name}", styles["Normal"]))
    story.append(Paragraph("Consultant Radiology Review", styles["Normal"]))
    
                                   
    story.append(Spacer(1, 0.4 * inch))
    
    disclaimer = (
        "<i><b>DISCLAIMER:</b> This report is AI-generated to support clinical review and documentation. "
        "It does not replace interpretation by a qualified radiologist or treating physician. "
        "Final clinical decisions should always be based on the patient history, examination, and full diagnostic context.</i>"
    )
    story.append(Paragraph(disclaimer, ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.red)))
    
    doc.build(story, onFirstPage=_draw_pdf_page, onLaterPages=_draw_pdf_page)
    buffer.seek(0)
    return buffer.getvalue()
