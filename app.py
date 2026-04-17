from __future__ import annotations

import os

# Set environment variables BEFORE any imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from medical_report_generator import (
    generate_medical_report_content,
    create_hospital_report_pdf,
)
from cv_model import (
    generate_gradcam_heatmap,
    overlay_heatmap_on_image,
    predict_xray,
    preprocess_uploaded_xray,
    load_pretrained_cv_model,
    validate_chest_xray,
)
from app_utils import (
    ASSETS_DIR,
    HISTORY_PATH,
    MODELS_DIR,
    append_history,
    clear_history,
    ensure_directories,
    get_history,
    get_image_as_base64,
    get_local_now,
    update_history_with_drive_url,
)
from google_drive_manager import GoogleDriveManager


st.set_page_config(
    page_title="AarogyaVeda",
    page_icon="H",
    layout="wide",
)


def _get_runtime_config(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value not in (None, ""):
        return str(value)
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


SHOW_REPORT_DEBUG = _get_runtime_config("SHOW_REPORT_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}

ensure_directories()


@st.cache_resource(show_spinner=False)
def get_cached_cv_model(
    model_path_str: str,
    model_mtime: float,
):
    model_path = Path(model_path_str)
    return load_pretrained_cv_model(model_path), {}


def build_architecture_figure(theme_mode: str) -> go.Figure:
    if theme_mode == "Dark":
        box_fill = "rgba(59, 47, 37, 0.95)"
        box_line = "#ffd166"
        text_color = "#f5ede0"
        arrow_color = "#ffd166"
    else:
        box_fill = "rgba(245, 250, 255, 0.96)"
        box_line = "#2d7a9e"
        text_color = "#1d3557"
        arrow_color = "#2d7a9e"

    fig = go.Figure()

    boxes = [
        (4, 82, 22, 94, "Upload X-ray\n(JPG/PNG)"),
        (24, 82, 42, 94, "Image Processing\nResize + Normalize"),
        (44, 82, 62, 94, "CV Pipeline\nLoad ResNet50 Model"),
        (64, 82, 82, 94, "Prediction\nNormal / Pneumonia"),
        (84, 82, 98, 94, "Confidence\n+ Grad-CAM Map"),
        (12, 58, 36, 70, "Image-Assisted Report\nX-ray + Grad-CAM Context"),
        (40, 58, 62, 70, "Hospital PDF\nPrepare + Preview"),
        (66, 58, 86, 70, "Save Report\nGoogle Drive Link"),
        (16, 34, 40, 46, "Adaptive Analytics\nDay/Week/Month"),
        (44, 34, 68, 46, "History Archive\nCSV + Drive Links"),
        (72, 34, 96, 46, "Dashboard Tabs\nHome/Imaging/Insights/History"),
    ]

    for x0, y0, x1, y1, label in boxes:
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color=box_line, width=2),
            fillcolor=box_fill,
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=label.replace("\\n", "<br>"),
            showarrow=False,
            font=dict(size=14, color=text_color),
            align="center",
        )

    arrows = [
        ((22, 88), (24, 88)),
        ((42, 88), (44, 88)),
        ((62, 88), (64, 88)),
        ((82, 88), (84, 88)),
        ((72, 82), (24, 70)),
        ((88, 82), (51, 70)),
        ((62, 64), (66, 64)),
        ((24, 58), (24, 46)),
        ((51, 58), (56, 46)),
        ((76, 58), (84, 46)),
    ]

    for (ax, ay), (x, y) in arrows:
        fig.add_annotation(
            x=x,
            y=y,
            ax=ax,
            ay=ay,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=arrow_color,
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(range=[20, 100], visible=False),
        height=680,
    )

    return fig


def render_home_metric(label: str, value: str) -> None:
    st.markdown(f"<div class=\"home-metric-label\">{label}</div><div class=\"home-metric-value\">{value}</div>", unsafe_allow_html=True)


def get_brand_logo_src() -> str:
    logo_candidates = [
        Path("assets/aarogyaveda_hero_banner.svg"),
        Path("Logo.png"),
        Path("AarogyaVeda logo landscape.png"),
        Path("AarogyaVeda_logo_landscape.png"),
        Path("AarogyaVeda logo.png"),
        Path("AarogyaVeda Sign.png"),
    ]
    for logo_path in logo_candidates:
        if logo_path.exists():
            encoded = get_image_as_base64(logo_path)
            mime = "image/svg+xml" if logo_path.suffix.lower() == ".svg" else "image/png"
            return f"data:{mime};base64,{encoded}"
    return ""


def apply_custom_css(theme_mode: str = "Light") -> None:
    bg_path = ASSETS_DIR / "background.svg"
    bg_data = get_image_as_base64(bg_path) if bg_path.exists() else ""

    if theme_mode == "Dark":
        primary = "#ffd166"
        card_bg = "rgba(45, 35, 25, 0.92)"
        text_color = "#f5ede0"
        header_grad = "linear-gradient(120deg, #8b6437 0%, #b8956a 100%)"
        background_style = "linear-gradient(135deg, #1f1510 0%, #3d2817 48%, #6b4e3d 100%)"
        tab_bg = "rgba(255, 209, 102, 0.22)"
        tab_text = "#fff6e8"
        control_bg = "#3b2f25"
        control_text = "#f0e6d8"
        control_border = "rgba(255, 209, 102, 0.5)"
        accent_bg = "#2563eb"
        accent_bg_soft = "rgba(59, 130, 246, 0.32)"
        accent_text = "#e0edff"
        header_bg = "rgba(50, 40, 30, 0.96)"
        metric_bg = "rgba(55, 45, 35, 0.95)"
        metric_text = "#f0e6d8"
    else:
        primary = "#b26a2a"
        card_bg = "rgba(255, 252, 245, 0.96)"
        text_color = "#2f1d0f"
        header_grad = "linear-gradient(120deg, #ffe8cc 0%, #ffd8a8 55%, #ffc078 100%)"
        background_style = "linear-gradient(140deg, #fff7ed 0%, #ffedd5 38%, #fde68a 100%)"
        tab_bg = "rgba(255, 248, 235, 0.95)"
        tab_text = "#4a2b12"
        control_bg = "#fffaf0"
        control_text = "#2f1d0f"
        control_border = "rgba(137, 80, 33, 0.34)"
        accent_bg = "#2563eb"
        accent_bg_soft = "rgba(37, 99, 235, 0.16)"
        accent_text = "#0b1f44"
        header_bg = "rgba(255, 247, 237, 0.95)"
        metric_bg = "rgba(255, 250, 240, 0.98)"
        metric_text = "#2f1d0f"

    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: linear-gradient(140deg, rgba(255, 255, 255, 0.20), rgba(255, 255, 255, 0.02)), url("data:image/svg+xml;base64,{bg_data}"), {background_style};
                background-size: cover;
                background-attachment: fixed;
            }}
            .block-container {{
                padding-top: 1.2rem;
            }}
            section.main h1,
            section.main h2,
            section.main h3,
            section.main h4,
            section.main p,
            section.main li,
            section.main label {{
                color: {text_color} !important;
            }}
            .hero-title {{
                position: relative;
                border-radius: 16px;
                padding: 18px 24px 22px 24px;
                background: {header_grad};
                border: 1px solid rgba(137, 80, 33, 0.35);
                box-shadow: 0 8px 24px rgba(0,0,0,0.12);
                margin-top: 80px;
                margin-bottom: 20px;
            }}
            .hero-logo {{
                display: block;
                width: 100%;
                height: 190px;
                object-fit: fill;
                object-position: center center;
                margin: 0 auto 16px auto;
                border-radius: 12px;
            }}
            .hero-copy {{
                position: relative;
                z-index: 1;
            }}
            .metric-card {{
                border-radius: 14px;
                background: {metric_bg} !important;
                border: 1px solid rgba(137, 80, 33, 0.26);
                padding: 12px;
                backdrop-filter: blur(6px);
            }}
            .metric-card [data-testid="metric-container"] {{
                background: transparent !important;
            }}
            .metric-card [data-testid="metric-container"] * {{
                color: {metric_text} !important;
            }}
            .metric-card [data-testid="stMetricSparkline"] {{
                display: none !important;
                height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
            }}
            .home-metric-label {{
                color: {metric_text};
                font-size: 1.05rem;
                font-weight: 500;
                margin-bottom: 0.35rem;
            }}
            .home-metric-value {{
                color: {metric_text};
                font-size: 2.5rem;
                line-height: 1.1;
                font-weight: 700;
            }}
            .risk-pill {{
                padding: 10px 16px;
                border-radius: 999px;
                color: white;
                font-weight: 700;
                display: inline-block;
                margin-top: 8px;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.4rem;
            }}
            .stTabs [data-baseweb="tab"] {{
                border-radius: 12px;
                background-color: {tab_bg};
                border: 1px solid rgba(137, 80, 33, 0.24);
                padding: 8px 14px;
                color: {tab_text} !important;
            }}
            .stButton > button {{
                border-radius: 10px;
                border: 1px solid {primary};
                font-weight: 600;
                color: {control_text} !important;
                background: {control_bg} !important;
            }}
            div[role="radiogroup"] {{
                gap: 0.35rem;
                flex-wrap: wrap;
            }}
            div[role="radiogroup"] > label {{
                border: 1px solid rgba(137, 80, 33, 0.32);
                background: {tab_bg};
                border-radius: 12px;
                padding: 6px 10px;
            }}
            div[role="radiogroup"] [data-baseweb="radio"] div[aria-checked="true"] > div:first-child {{
                background: {accent_bg} !important;
                border-color: {accent_bg} !important;
            }}
            div[role="radiogroup"] [data-baseweb="radio"]:hover {{
                background: {accent_bg_soft} !important;
                border-radius: 10px;
            }}
            .stSelectbox div[data-baseweb="select"] > div,
            .stMultiselect div[data-baseweb="select"] > div,
            .stNumberInput input,
            .stTextInput input,
            .stTextArea textarea,
            .stDateInput input,
            .stTimeInput input {{
                background: {control_bg} !important;
                color: {control_text} !important;
                border: 1px solid {control_border} !important;
            }}
            .stSelectbox div[data-baseweb="select"] input,
            .stMultiselect div[data-baseweb="select"] input,
            .stNumberInput input,
            .stTextInput input,
            .stTextArea textarea {{
                color: {control_text} !important;
                -webkit-text-fill-color: {control_text} !important;
                caret-color: {control_text} !important;
            }}
            .stSlider [data-baseweb="slider"] * {{
                color: {control_text} !important;
            }}
            [data-testid="stFileUploaderDropzone"] {{
                background: {control_bg} !important;
                border: 2px dashed {control_border} !important;
                border-radius: 8px !important;
                padding: 20px !important;
                text-align: center !important;
            }}
            [data-testid="stFileUploaderDropzone"] * {{
                color: {control_text} !important;
            }}
            [data-testid="stFileUploaderDropzone"] [kind="secondary"] {{
                background: {control_bg} !important;
                color: {control_text} !important;
                border: 1px solid {control_border} !important;
                margin-top: 12px !important;
            }}
            [data-baseweb="popover"] * {{
                color: {control_text} !important;
            }}
            [role="listbox"] {{
                background: {control_bg} !important;
            }}
            [role="option"] {{
                background: {control_bg} !important;
                color: {control_text} !important;
            }}
            [role="option"][aria-selected="true"],
            [role="option"]:hover {{
                background: {accent_bg_soft} !important;
                color: {accent_text} !important;
            }}
            .stSelectbox div[data-baseweb="select"] svg,
            .stMultiselect div[data-baseweb="select"] svg {{
                fill: {control_text} !important;
                color: {control_text} !important;
            }}
            [data-testid="stFileUploaderDropzone"] > div:first-child {{
                background: transparent !important;
            }}
            [data-testid="stHeader"] {{
                background: transparent !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    st.session_state.setdefault("theme_mode", "Light")
    st.session_state.setdefault("images_processed", 0)
    st.session_state.setdefault("reports_analyzed", 0)
    st.session_state.setdefault("report_key", "")
    st.session_state.setdefault("report_text", "")
    st.session_state.setdefault("history_logged_keys", [])
    st.session_state.setdefault("current_upload_key", "")
    st.session_state.setdefault("last_pdf_bytes", None)
    st.session_state.setdefault("last_pdf_filename", "")
    st.session_state.setdefault("_oauth_code_processed", False)
    st.session_state.setdefault("_oauth_authenticated", False)


init_session_state()


def restore_history_archive_from_drive() -> None:
    drive_manager = st.session_state.get("drive_manager")
    if not drive_manager or not getattr(drive_manager, "is_authenticated", False):
        return

    if HISTORY_PATH.exists() and HISTORY_PATH.stat().st_size > 0:
        return

    try:
        csv_bytes = drive_manager.download_file_by_name(HISTORY_PATH.name)
        if csv_bytes:
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            HISTORY_PATH.write_bytes(csv_bytes)
    except Exception:
        pass


def sync_history_archive_to_drive() -> None:
    drive_manager = st.session_state.get("drive_manager")
    if not drive_manager or not getattr(drive_manager, "is_authenticated", False):
        return

    if not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0:
        return

    try:
        from io import BytesIO

        with open(HISTORY_PATH, "rb") as file:
            history_bytes = BytesIO(file.read())
        drive_manager.upload_file(history_bytes, HISTORY_PATH.name, "text/csv", replace_existing=True)
    except Exception:
        pass

# Handle OAuth callback from Google Drive authorization
auth_code = None
try:
    auth_code = st.query_params.get("code")
except Exception:
    try:
        auth_code = st.experimental_get_query_params().get("code")
    except Exception:
        auth_code = None

# Track if we've already processed this specific auth code
_oauth_code_processed = st.session_state.get("_oauth_code_processed", False)

# Initialize drive manager once to handle loading cached credentials
if not st.session_state.get("drive_manager"):
    try:
        st.session_state["drive_manager"] = GoogleDriveManager()
    except FileNotFoundError:
        st.session_state["drive_manager"] = None
    except Exception:
        st.session_state["drive_manager"] = None

# Only process auth code once, and only if we don't already have a valid token
_already_authenticated = "_drive_token_json" in st.session_state or (
    st.session_state.get("drive_manager") and getattr(st.session_state["drive_manager"], "is_authenticated", False)
)

if auth_code and not _oauth_code_processed and not _already_authenticated:
    st.session_state["_oauth_code_processed"] = True
    # Create fresh drive manager which will process the auth code
    try:
        st.session_state["drive_manager"] = GoogleDriveManager()
    except FileNotFoundError:
        st.session_state["drive_manager"] = None
    except Exception:
        st.session_state["drive_manager"] = None
    
    # Clear query params and rerun to remove the auth code from URL
    try:
        st.query_params.clear()
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass
    st.rerun()

# Check if this is a page showing successful OAuth completion
_oauth_authenticated = st.session_state.get("_oauth_authenticated", False)
if _oauth_authenticated:
    st.session_state["_oauth_authenticated"] = False
    # Show success message 
    st.success("✅ Google Drive Authentication Successful!")
    st.info("""
    Your Google Drive is now connected and authenticated. 
    
    **Next steps:**
    1. You can **close this tab** or go back to the original one
    2. The authorization is saved for this session
    3. **Reload the page** if the original tab still shows "not connected"
    
    You only need to authorize once per session!
    """)

apply_custom_css(st.session_state["theme_mode"])
restore_history_archive_from_drive()
history_df = get_history()
logo_src = get_brand_logo_src()

st.markdown("")

menu_display = {
    "HOME": "TAB 1 HOME",
    "CLINICAL IMAGING ANALYSIS": "TAB 2 MEDICAL IMAGE DETECTION",
    "INSIGHTS DASHBOARD": "TAB 3 DATA ANALYTICS",
    "HISTORY": "TAB 4 PREDICTION HISTORY",
}

selected_label = st.radio(
    "Navigation",
    options=list(menu_display.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
selected_tab = menu_display[selected_label]


if selected_tab == "TAB 1 HOME":
    hero_logo_html = f'<img src="{logo_src}" class="hero-logo" alt="AarogyaVeda Banner" />' if logo_src else ""
    st.markdown(
        f"""
        <div class="hero-title">
            {hero_logo_html}
            <div class="hero-copy">
                <h1 style="margin:0;">AarogyaVeda</h1>
                <p style="margin:6px 0 0 0;">AI-Powered Clinical Imaging Assistant</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Clinical Intelligence Overview")
    st.write(
        "This platform helps clinicians analyze chest X-rays faster, generate professional report drafts, and export polished PDF documents for care, review, and archival."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        render_home_metric(
            "Reports generated",
            str(max(st.session_state["reports_analyzed"], len(history_df[history_df["prediction_type"] == "X-ray"]) if not history_df.empty else 0)),
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        render_home_metric(
            "Images analyzed",
            str(max(st.session_state["images_processed"], len(history_df[history_df["prediction_type"] == "X-ray"]) if not history_df.empty else 0)),
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        render_home_metric("History records", str(len(history_df)))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Architecture")
    st.plotly_chart(
        build_architecture_figure(st.session_state["theme_mode"]),
        width='stretch',
        config={"displayModeBar": False},
    )


elif selected_tab == "TAB 2 MEDICAL IMAGE DETECTION":
    st.subheader("Clinical Imaging Analysis & Report Generation")
    st.caption("Upload a chest X-ray, review AI-supported findings, and download a professional medical report as PDF")

    drive_manager = st.session_state.get("drive_manager")
    
    is_authenticated = bool(drive_manager and getattr(drive_manager, "is_authenticated", False))
    auth_mode = getattr(drive_manager, "auth_mode", None) if drive_manager else None

    if not is_authenticated and drive_manager and getattr(drive_manager, "auth_url", None):
        auth_url = drive_manager.auth_url
        st.info("Google Drive OAuth is enabled. If needed, authorize using the link below.")
        st.markdown(f"[Authorize Google Drive]({auth_url})")
    elif not is_authenticated:
        st.warning("⚠️ Google Drive is not configured.")
        if drive_manager and getattr(drive_manager, "last_error", None):
            st.error(f"Drive setup error: {drive_manager.last_error}")
        with st.expander("Setup (recommended: personal OAuth refresh token)"):
            st.markdown(
                """
                Add these in Streamlit secrets:

            1. `GOOGLE_CLIENT_ID`
            2. `GOOGLE_CLIENT_SECRET`
            3. `GOOGLE_REFRESH_TOKEN`
            4. `GOOGLE_DRIVE_FOLDER_ID` with your target Google Drive folder id

            This method uses your personal Google account and does not require service accounts.

                Optional fallback:

                - Set `ENABLE_DRIVE_OAUTH = true` only if you explicitly want browser OAuth.
                """
            )

    backbone = "ResNet50"
    image_size = 224
    model_file = MODELS_DIR / f"cv_{backbone.lower()}_{image_size}.keras"
    selected_cv_key = f"{backbone}_{image_size}"

    if st.session_state.get("cv_model_key") == selected_cv_key and not model_file.exists():
        st.session_state.pop("cv_model", None)
        st.session_state.pop("cv_model_name", None)
        st.session_state.pop("cv_scores", None)
        st.session_state.pop("cv_model_key", None)

    if st.session_state.get("cv_model_key") != selected_cv_key and model_file.exists():
        with st.spinner("Loading saved CV model..."):
            try:
                cv_model, cv_scores = get_cached_cv_model(
                    str(model_file),
                    model_file.stat().st_mtime,
                )
                st.session_state["cv_model"] = cv_model
                st.session_state["cv_model_name"] = backbone
                st.session_state["cv_model_key"] = selected_cv_key
                if cv_scores:
                    st.session_state["cv_scores"] = cv_scores
                else:
                    st.session_state.pop("cv_scores", None)
            except Exception as exc:
                st.warning(f"Saved CV model could not be loaded in this environment: {str(exc)[:180]}")
                st.session_state.pop("cv_model", None)
                st.session_state.pop("cv_model_name", None)
                st.session_state.pop("cv_scores", None)
                st.session_state.pop("cv_model_key", None)

    if not model_file.exists():
        st.info("No saved CV model found in the models folder. The app can only run inference after the pre-trained model is available.")

    uploaded = st.file_uploader("Upload chest X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded is not None and "cv_model" in st.session_state and st.session_state.get("cv_model_key") == selected_cv_key:
        from PIL import Image

        upload_key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("current_upload_key") != upload_key:
            image = Image.open(uploaded).convert("RGB")
            
            # Validate if image is a chest X-ray
            is_valid_xray, validation_message = validate_chest_xray(image)
            
            if not is_valid_xray:
                st.error(validation_message)
                st.session_state["current_upload_key"] = upload_key
                st.session_state.pop("xray_image", None)
                st.session_state.pop("xray_input_arr", None)
                st.session_state.pop("xray_prediction", None)
                st.session_state.pop("xray_heatmap", None)
                st.session_state.pop("xray_heatmap_context", None)
                st.session_state.pop("xray_gradcam_overlay", None)
                st.stop()
            else:
                input_arr = preprocess_uploaded_xray(image, image_size=(image_size, image_size))
                pred = predict_xray(st.session_state["cv_model"], input_arr, threshold=0.70)

                heatmap = None
                heatmap_context = "Heatmap not available."
                try:
                    image_tensor = np.array(input_arr[0], dtype=float)
                    gray = np.mean(image_tensor, axis=2)
                    mean_intensity = float(np.mean(gray))
                    contrast_level = float(np.std(gray))
                    left_mean = float(np.mean(gray[:, : gray.shape[1] // 2]))
                    right_mean = float(np.mean(gray[:, gray.shape[1] // 2 :]))
                    side_bias = "left-dominant" if left_mean > right_mean + 0.02 else ("right-dominant" if right_mean > left_mean + 0.02 else "balanced")

                    brightness_desc = "hyperlucent" if mean_intensity > 0.62 else ("low-lucency" if mean_intensity < 0.38 else "mid-lucency")
                    contrast_desc = "high-contrast" if contrast_level > 0.22 else ("low-contrast" if contrast_level < 0.12 else "moderate-contrast")

                    heatmap = generate_gradcam_heatmap(st.session_state["cv_model"], input_arr)
                    hm = np.array(heatmap, dtype=float)
                    ys, xs = np.indices(hm.shape)
                    total = float(np.sum(hm))
                    if total > 0:
                        cx = float(np.sum(hm * xs) / total) / max(hm.shape[1] - 1, 1)
                        cy = float(np.sum(hm * ys) / total) / max(hm.shape[0] - 1, 1)
                        horiz = "left" if cx < 0.4 else ("right" if cx > 0.6 else "central")
                        vert = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "mid")
                        region = f"{vert}-{horiz}"
                        max_attention = float(np.max(hm))
                        intensity = "strong" if max_attention > np.percentile(hm, 80) else "moderate" if max_attention > np.percentile(hm, 50) else "subtle"
                        assessment = pred["predicted_class"].lower()

                        if assessment == "pneumonia":
                            heatmap_context = (
                                f"The model identified disease indicators with {intensity} attention in the {region} lung zone. "
                                f"This {region} localization is consistent with focal pneumonic consolidation or infiltrate. "
                                f"Image texture profile is {brightness_desc}, {contrast_desc}, and {side_bias} across hemithoraces. "
                                f"Key clinical features to analyze in this region: opacity density, extent of involvement, signs of complications. "
                                f"Consider clinical correlation with patient symptoms and physical examination findings."
                            )
                        else:
                            heatmap_context = (
                                f"The model identified normal lung features with {intensity} attention in the {region} lung zone. "
                                f"Image texture profile is {brightness_desc}, {contrast_desc}, with a {side_bias} intensity distribution. "
                                f"The {region} region shows expected radiographic patterns for healthy lungs. "
                                f"Overall lung fields appear clear with normal cardiac silhouette and mediastinal contours. "
                                f"No acute cardiopulmonary process identified."
                            )
                    else:
                        heatmap_context = (
                            f"The model identified {pred['predicted_class'].lower()} features distributed across the lung fields. "
                            f"Image texture profile is {brightness_desc}, {contrast_desc}, and {side_bias}. "
                            f"Assess the entire image for comprehensive radiographic findings."
                        )
                except Exception:
                    heatmap = None
                    heatmap_context = "Detailed heatmap analysis not available. Assess the entire image for clinical findings."

                st.session_state["current_upload_key"] = upload_key
                st.session_state["current_upload_name"] = uploaded.name
                st.session_state["xray_image"] = image
                st.session_state["xray_input_arr"] = input_arr
                st.session_state["xray_prediction"] = pred
                st.session_state["xray_heatmap"] = heatmap
                st.session_state["xray_heatmap_context"] = heatmap_context
                try:
                    if heatmap is not None:
                        from PIL import Image as PILImage
                        gradcam_array = overlay_heatmap_on_image(image, heatmap)
                        st.session_state["xray_gradcam_overlay"] = PILImage.fromarray(gradcam_array)
                    else:
                        st.session_state["xray_gradcam_overlay"] = None
                except Exception:
                    st.session_state["xray_gradcam_overlay"] = None
                st.session_state.pop("last_report_content", None)
                st.session_state.pop("last_prediction", None)
                st.session_state.pop("last_patient_name", None)
                st.session_state.pop("last_patient_id", None)
                st.session_state["last_pdf_bytes"] = None
                st.session_state["last_pdf_filename"] = ""

    image = st.session_state.get("xray_image")
    input_arr = st.session_state.get("xray_input_arr")
    pred = st.session_state.get("xray_prediction")
    heatmap = st.session_state.get("xray_heatmap")
    gradcam_overlay = st.session_state.get("xray_gradcam_overlay")
    heatmap_context = st.session_state.get("xray_heatmap_context", "Detailed heatmap analysis not available. Assess the entire image for clinical findings.")

    if image is not None and input_arr is not None and pred is not None:
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.image(image, caption="Uploaded Chest X-ray", width="stretch")
        with result_col2:
            if pred["predicted_class"] == "PNEUMONIA":
                st.error(f"Assessment: {pred['predicted_class']}")
            else:
                st.success(f"Assessment: {pred['predicted_class']}")
            st.metric("Pneumonia Probability", f"{pred['pneumonia_probability'] * 100:.2f}%")

        with st.expander("View GradCAM Heatmap"):
            try:
                if heatmap is None:
                    heatmap = generate_gradcam_heatmap(st.session_state["cv_model"], input_arr)
                    st.session_state["xray_heatmap"] = heatmap
                if gradcam_overlay is None and heatmap is not None:
                    from PIL import Image as PILImage
                    gradcam_array = overlay_heatmap_on_image(image, heatmap)
                    gradcam_overlay = PILImage.fromarray(gradcam_array)
                    st.session_state["xray_gradcam_overlay"] = gradcam_overlay
                if gradcam_overlay is not None:
                    st.image(gradcam_overlay, caption="AI-Identified Focus Areas", width="stretch")
            except Exception:
                st.warning("Could not generate GradCAM visualization.")

        st.divider()
        st.markdown("### Generate Clinical Report")

        with st.form("clinical_report_form", clear_on_submit=False):
            input_col1, input_col2 = st.columns(2)
            with input_col1:
                patient_id_input = st.text_input("Patient ID", key="patient_id_input", placeholder="Enter patient ID")
            with input_col2:
                patient_name_input = st.text_input("Patient Name", key="patient_name_input", placeholder="Enter patient name")

            generate_report = st.form_submit_button("Generate Clinical Report")

        if generate_report:
            if not patient_id_input.strip() or not patient_name_input.strip():
                st.warning("Patient ID and Patient Name are required to generate a report.")
            else:
                with st.spinner("Generating the clinical report..."):
                    try:
                        try:
                            report_content = generate_medical_report_content(
                                prediction_label=pred["predicted_class"],
                                confidence=pred["confidence"],
                                patient_name=patient_name_input,
                                patient_id=patient_id_input,
                                case_context=heatmap_context,
                                uploaded_image=image,
                                gradcam_image=gradcam_overlay,
                            )
                        except TypeError as sig_err:
                            err_msg = str(sig_err)
                            if "unexpected keyword argument 'gradcam_image'" in err_msg:
                                try:
                                    report_content = generate_medical_report_content(
                                        prediction_label=pred["predicted_class"],
                                        confidence=pred["confidence"],
                                        patient_name=patient_name_input,
                                        patient_id=patient_id_input,
                                        case_context=heatmap_context,
                                        uploaded_image=image,
                                    )
                                except TypeError as inner_sig_err:
                                    if "unexpected keyword argument 'uploaded_image'" not in str(inner_sig_err):
                                        raise
                                    report_content = generate_medical_report_content(
                                        prediction_label=pred["predicted_class"],
                                        confidence=pred["confidence"],
                                        patient_name=patient_name_input,
                                        patient_id=patient_id_input,
                                        case_context=heatmap_context,
                                    )
                            elif "unexpected keyword argument 'uploaded_image'" in err_msg:
                                report_content = generate_medical_report_content(
                                    prediction_label=pred["predicted_class"],
                                    confidence=pred["confidence"],
                                    patient_name=patient_name_input,
                                    patient_id=patient_id_input,
                                    case_context=heatmap_context,
                                )
                            else:
                                raise
                        st.session_state["last_report_content"] = report_content
                        st.session_state["last_prediction"] = pred
                        st.session_state["last_patient_name"] = patient_name_input
                        st.session_state["last_patient_id"] = patient_id_input
                        st.success("Clinical report generated successfully!")
                        generation_mode = report_content.get("generation_mode", "unknown")
                        model_used = report_content.get("model_used", "unknown")
                        caption_source = report_content.get("caption_source", "")
                        images_analyzed = report_content.get("images_analyzed", "unknown")
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        st.info("Make sure HF_TOKEN is set in .env or run `hf auth login`")

        if "last_report_content" in st.session_state:
            st.markdown("#### Report Preview")

            report_tabs = st.tabs(["Findings", "Impression", "Precautions"])

            with report_tabs[0]:
                st.markdown(st.session_state["last_report_content"]["findings"])

            with report_tabs[1]:
                st.markdown(st.session_state["last_report_content"]["impression"])

            with report_tabs[2]:
                st.markdown(st.session_state["last_report_content"]["precautions"])

            st.divider()

            download_col1, download_col2 = st.columns([2, 1])
            with download_col1:
                doctor_name = st.text_input("Physician Name (for report)", value="Dr. Medical AI", key="doctor_name")
            with download_col2:
                hospital_name = st.text_input("Hospital/Clinic Name", value="AarogyaVeda Medical Center", key="hospital_name")

            if st.button("Prepare Clinical PDF", width='stretch', type="primary"):
                try:
                    pdf_bytes = create_hospital_report_pdf(
                        prediction_label=st.session_state["last_prediction"]["predicted_class"],
                        confidence=st.session_state["last_prediction"]["confidence"],
                        report_content=st.session_state["last_report_content"],
                        patient_name=st.session_state["last_patient_name"],
                        patient_id=st.session_state["last_patient_id"],
                        doctor_name=doctor_name,
                        hospital_name=hospital_name,
                    )

                    filename = f"XRay_Report_{st.session_state['last_patient_name'].replace(' ', '_')}_{get_local_now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.session_state["last_pdf_bytes"] = pdf_bytes
                    st.session_state["last_pdf_filename"] = filename
                    st.success("Clinical PDF prepared. Click Save Report below.")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

            if st.session_state.get("last_pdf_bytes"):
                record_key = f"{st.session_state.get('current_upload_key', '')}:{st.session_state['last_prediction']['predicted_class']}:{st.session_state['last_prediction']['confidence']:.6f}"
                if st.session_state.get("drive_manager"):
                    if st.button("Save Report", width='stretch', key="save_report"):
                        pdf_buffer = BytesIO(st.session_state["last_pdf_bytes"])
                        upload_result = st.session_state["drive_manager"].upload_pdf(
                            pdf_buffer, st.session_state.get("last_pdf_filename", "report.pdf")
                        )

                        if upload_result["success"]:
                            st.success("Saved to Google Drive.")
                            st.session_state["last_drive_url"] = upload_result.get("view_url", upload_result["download_url"])

                            if record_key not in st.session_state["history_logged_keys"]:
                                append_history(
                                    prediction_type="X-ray",
                                    model_name=st.session_state.get("cv_model_name", backbone),
                                    patient_id=st.session_state.get("last_patient_id", "N/A"),
                                    patient_name=st.session_state.get("last_patient_name", "Patient"),
                                    input_summary=f"Image: {st.session_state.get('current_upload_name', 'uploaded_xray')}",
                                    predicted_label=st.session_state["last_prediction"]["predicted_class"],
                                    risk_probability=st.session_state["last_prediction"]["pneumonia_probability"],
                                    confidence=st.session_state["last_prediction"]["confidence"],
                                    drive_url=upload_result.get("view_url", upload_result["download_url"]),
                                    history_key=record_key,
                                )
                                sync_history_archive_to_drive()
                                st.session_state["images_processed"] += 1
                                st.session_state["reports_analyzed"] += 1
                                st.session_state["history_logged_keys"].append(record_key)
                            else:
                                update_history_with_drive_url(
                                    record_key,
                                    upload_result.get("view_url", upload_result["download_url"]),
                                )
                                sync_history_archive_to_drive()

                            st.info(
                                f"📄 [View on Google Drive]({upload_result.get('view_url', upload_result['download_url'])})"
                            )
                        else:
                            st.error(f"Failed to save: {upload_result['error']}")
                else:
                    st.warning("Google Drive is not connected.")
                    if drive_manager and getattr(drive_manager, "last_error", None):
                        st.error(f"Drive setup error: {drive_manager.last_error}")


elif selected_tab == "TAB 3 DATA ANALYTICS":
    st.subheader("Insights Dashboard")

    live_history = get_history()
    if live_history.empty:
        st.info("No analysis data yet. Start by uploading X-rays to generate reports.")
    else:
                                                       
        diagnosis_col = None
        if "predicted_label" in live_history.columns:
            diagnosis_col = "predicted_label"
        elif "prediction_label" in live_history.columns:
            diagnosis_col = "prediction_label"

        if diagnosis_col:
            pred_counts = live_history[diagnosis_col].dropna().astype(str).value_counts().reset_index()
            pred_counts.columns = ["Diagnosis", "Count"]
            pred_fig = px.pie(pred_counts, names="Diagnosis", values="Count", title="Pneumonia vs Normal Distribution")
            st.plotly_chart(pred_fig, width='stretch')

                              
        if "timestamp" in live_history.columns:
            timeline_df = live_history.copy()
            timeline_df["timestamp"] = pd.to_datetime(timeline_df["timestamp"], errors="coerce")
            timeline_df = timeline_df.dropna(subset=["timestamp"])
            if not timeline_df.empty:
                timeline_df = timeline_df.sort_values("timestamp")
                start_date = timeline_df["timestamp"].min().normalize()
                end_date = timeline_df["timestamp"].max().normalize()
                day_span = max((end_date - start_date).days + 1, 1)
                unique_days = timeline_df["timestamp"].dt.normalize().nunique()

                if day_span <= 14 and unique_days <= 14:
                    timeline_df["bucket"] = timeline_df["timestamp"].dt.normalize()
                    timeline_df_grouped = timeline_df.groupby("bucket", as_index=False).size().rename(columns={"size": "Analyses"})
                    chart_title = "Analyses by Day"
                    x_title = "Day"
                    timeline_df_grouped["BucketLabel"] = timeline_df_grouped["bucket"].dt.strftime("%d %b")
                    bar_width = 0.5
                elif day_span <= 120:
                    timeline_df["bucket"] = timeline_df["timestamp"].dt.to_period("W-MON").dt.start_time
                    timeline_df_grouped = timeline_df.groupby("bucket", as_index=False).size().rename(columns={"size": "Analyses"})
                    chart_title = "Analyses by Week"
                    x_title = "Week"
                    timeline_df_grouped["BucketLabel"] = timeline_df_grouped["bucket"].dt.strftime("Week of %d %b")
                    bar_width = 0.5
                else:
                    timeline_df["bucket"] = timeline_df["timestamp"].dt.to_period("M").dt.start_time
                    timeline_df_grouped = timeline_df.groupby("bucket", as_index=False).size().rename(columns={"size": "Analyses"})
                    chart_title = "Analyses by Month"
                    x_title = "Month"
                    timeline_df_grouped["BucketLabel"] = timeline_df_grouped["bucket"].dt.strftime("%b %Y")
                    bar_width = 0.5

                timeline_fig = px.bar(
                    timeline_df_grouped,
                    x="BucketLabel",
                    y="Analyses",
                    title=chart_title,
                    hover_data={"bucket": True, "BucketLabel": False},
                )
                timeline_fig.update_traces(width=bar_width)
                timeline_fig.update_xaxes(title=x_title, type="category", tickangle=0)
                st.plotly_chart(timeline_fig, width='stretch')

elif selected_tab == "TAB 4 PREDICTION HISTORY":
    st.subheader("Report Archive")

    if st.button("Reset history"):
        clear_history()
        sync_history_archive_to_drive()
        st.success("Archive cleared.")
        history_df = get_history()

    latest_history = get_history()
    if latest_history.empty:
        st.info("No analyses logged yet.")
    else:
        drive_lookup = {}
        if st.session_state.get("drive_manager"):
            try:
                drive_files = st.session_state["drive_manager"].get_file_list(limit=200)
                for file in drive_files:
                    drive_lookup[file.get("name", "")] = file.get("view_url") or file.get("download_url", "")
            except Exception:
                drive_lookup = {}

                                                           
        display_cols = ["patient_id", "patient_name", "timestamp", "predicted_label", "prediction_type"]
        
                                     
        display_cols = [col for col in display_cols if col in latest_history.columns]
        history_view = latest_history[display_cols].copy()
        
                                                    
        history_view["download_link"] = ""

        has_drive_url = "drive_url" in latest_history.columns

        for idx, row in history_view.iterrows():
            drive_url = ""
            if has_drive_url:
                raw_url = latest_history.iloc[idx].get("drive_url", "")
                if raw_url and str(raw_url).strip() and str(raw_url) != "nan":
                    drive_url = str(raw_url).strip()
                    if "drive.google.com/uc?export=download&id=" in drive_url:
                        file_id = drive_url.split("id=")[-1].split("&")[0]
                        drive_url = f"https://drive.google.com/file/d/{file_id}/view"

                                                                                             
            if not drive_url and drive_lookup:
                patient_token = str(row.get("patient_name", "")).strip().replace(" ", "_").lower()
                ts = str(row.get("timestamp", "")).strip()
                date_token = ""
                if ts:
                    date_token = ts.split(" ")[0].replace("-", "")

                for filename, url in drive_lookup.items():
                    lower_name = filename.lower()
                    if patient_token and patient_token in lower_name and (not date_token or date_token in lower_name):
                        drive_url = url
                        break

            history_view.at[idx, "download_link"] = drive_url
        
                                               
        history_view = history_view.rename(columns={
            "patient_id": "Patient ID",
            "patient_name": "Patient Name",
            "timestamp": "Date/Time",
            "predicted_label": "Diagnosis",
            "prediction_type": "Analysis Type",
            "download_link": "Download"
        })

        st.dataframe(
            history_view,
            width='stretch',
            column_config={
                "Download": st.column_config.LinkColumn(
                    "Download",
                    help="Open PDF report from Google Drive",
                    display_text="Open PDF",
                )
            },
        )
        
                      
        history_csv = latest_history.copy()
        if "history_key" in history_csv.columns:
            history_csv = history_csv.drop(columns=["history_key"])

        if "confidence" in history_csv.columns:
            history_csv = history_csv.drop(columns=["confidence"])

        if "risk_probability" in history_csv.columns:
            def _to_percent(value):
                try:
                    val = float(value)
                    pct = val * 100.0 if val <= 1.0 else val
                    return f"{pct:.2f}%"
                except (TypeError, ValueError):
                    return value

            history_csv["risk_probability"] = history_csv["risk_probability"].apply(_to_percent)
        
                                                   
        if "drive_url" in history_csv.columns:
            history_csv["drive_url"] = history_csv["drive_url"].apply(
                lambda url: f"https://drive.google.com/file/d/{url.split('id=')[-1].split('&')[0]}/view"
                if isinstance(url, str) and "drive.google.com/uc?export=download&id=" in url
                else url
            )
        
        csv_bytes = history_csv.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download archive as CSV",
            data=csv_bytes,
            file_name="xray_analysis_history.csv",
            mime="text/csv",
        )

