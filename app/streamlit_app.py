"""Streamlit web app: WasteWise - Waste Classification for Recycling.

Every image (upload or camera) goes through src.predict.predict_image(), which uses only the
Google Teachable Machine model. There is no fallback classifier.

UI: dark emerald theme (navbar, hero, side-by-side upload/result panels, category cards,
impact / about / FAQ sections, footer). All numbers shown come from the real model output,
the real class list and reports/metrics.json.
"""

import base64
import html
import io
import json
import logging
import sys
from pathlib import Path

# Make the project root importable when launched with `streamlit run app/streamlit_app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.predict import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    InvalidImageError,
    ModelLoadError,
    ModelNotFoundError,
    load_image,
    load_model_and_classes,
    predict_image,
)
from src.recycling_info import DISCLAIMER, get_info  # noqa: E402

APP_TITLE = "Waste Classification for Recycling"
BRAND = "WasteWise"
GROUP_ID = "BT_DL_09"
PROJECT_ID = "DL-009"
CLASS_LINE = "B.Tech Sem V, Section B"
MENTOR = "Sachin Raghav"
UNIVERSITY = "Invertis University"
MAX_UPLOAD_MB = 10
HIGH_CONFIDENCE = 0.90

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"{BRAND} | {APP_TITLE}",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------------------
COLOR_GREEN = "#2E7D4F"
COLOR_AMBER = "#F2A33A"
COLOR_RED = "#B5472A"
COLOR_BROWN = "#8A6A3B"
COLOR_GREY = "#6B7B72"
GLOW_GREEN = "#4ADE80"


def category_color(category: str) -> str:
    c = str(category).lower()
    if any(k in c for k in ("hazard", "e-waste", "electronic", "toxic")):
        return COLOR_RED
    if any(k in c for k in ("organic", "compost", "bio")):
        return COLOR_BROWN
    if any(k in c for k in ("landfill", "general", "non-recycl", "trash", "residual")):
        return COLOR_GREY
    return COLOR_GREEN


def confidence_color(confidence: float, threshold: float) -> str:
    if confidence >= HIGH_CONFIDENCE:
        return GLOW_GREEN
    if confidence >= threshold:
        return COLOR_AMBER
    return "#F0674A"


# --------------------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html { scroll-behavior: smooth; }
.stApp {
  font-family: 'Poppins', 'Segoe UI', system-ui, sans-serif;
  color: #E6F4EA;
  background:
    radial-gradient(1100px 560px at 85% -8%, rgba(16,185,129,.20), transparent 60%),
    radial-gradient(900px 520px at -5% 38%, rgba(34,197,94,.10), transparent 60%),
    linear-gradient(180deg, #07130F 0%, #0A1A14 60%, #06100C 100%);
}
.stApp [data-testid="stMarkdownContainer"] p { color: inherit; }

/* hide default Streamlit chrome */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

.block-container { max-width: 1280px; padding: 5.4rem 1.2rem 2rem !important; }
.anchor { scroll-margin-top: 84px; }

/* ---------- navbar ---------- */
.nav { position: fixed; top: 0; left: 0; right: 0; z-index: 1000; height: 64px;
       display: flex; align-items: center; justify-content: space-between; padding: 0 2rem;
       background: rgba(6,18,14,.90); backdrop-filter: blur(10px);
       border-bottom: 1px solid rgba(74,222,128,.18); }
.nav-brand { display: flex; align-items: center; gap: .55rem; font-weight: 800; font-size: 1.45rem; color: #fff; }
.nav-brand span { color: #4ADE80; }
.nav-links { display: flex; gap: .3rem; }
.nav-links a { color: #CFE8D8 !important; text-decoration: none !important; padding: .5rem 1rem;
               border-radius: 12px; font-weight: 600; font-size: .95rem; transition: background .2s; }
.nav-links a:hover, .nav-links a.active { background: rgba(34,197,94,.20); color: #fff !important; }
.nav-tag { font-size: .8rem; color: #CFE8D8; background: rgba(74,222,128,.14);
           border: 1px solid rgba(74,222,128,.3); padding: .3rem .8rem; border-radius: 999px; }

/* ---------- hero ---------- */
.hero { position: relative; overflow: hidden; border-radius: 22px; padding: 2rem 2.2rem; min-height: 240px;
        margin-bottom: 1.1rem; border: 1px solid rgba(74,222,128,.28);
        background:
          radial-gradient(480px 300px at 84% 45%, rgba(74,222,128,.38), transparent 65%),
          radial-gradient(260px 200px at 4% 90%, rgba(74,222,128,.22), transparent 70%),
          linear-gradient(120deg, #0B3A2A 0%, #0F5B3A 55%, #0B3A2A 100%);
        box-shadow: 0 0 44px rgba(34,197,94,.16) inset; }
.hero-chip { display: inline-flex; align-items: center; gap: .4rem; background: rgba(255,255,255,.12);
             padding: .3rem .85rem; border-radius: 10px; font-size: .85rem; font-weight: 600; color: #fff; }
.hero-title { font-size: clamp(2rem, 4.6vw, 3.3rem); font-weight: 800; line-height: 1.1;
              margin: .9rem 0 .4rem; color: #fff; }
.hero-title b { color: #4ADE80; }
.hero-sub { font-size: 1.1rem; color: #E3F4E8; max-width: 640px; }
.hero-badges { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: .6rem; }
.hero-badge { background: rgba(6,18,14,.55); border: 1px solid rgba(74,222,128,.25); color: #fff;
              padding: .4rem 1rem; border-radius: 12px; font-size: .9rem; font-weight: 600; }
.hero-globe { position: absolute; right: 8%; top: 50%; transform: translateY(-50%); width: 210px; height: 210px;
              border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 104px;
              background: radial-gradient(circle at 35% 30%, #3EE08A, #0E7A45 55%, #06402A);
              box-shadow: 0 0 70px rgba(74,222,128,.55), inset 0 0 30px rgba(0,0,0,.35); }
.hero-slogan { position: absolute; right: 2rem; top: 1.1rem; font-style: italic; font-weight: 600;
               font-size: 1.25rem; line-height: 1.25; color: #fff; transform: rotate(-6deg); text-align: left; }

/* ---------- panels (st.container with key) ---------- */
.st-key-upload_panel, .st-key-result_panel {
  background: linear-gradient(160deg, rgba(14,48,36,.88), rgba(8,26,20,.92));
  border: 1px solid rgba(74,222,128,.38); border-radius: 18px; padding: 1.1rem 1.2rem;
  box-shadow: 0 0 26px rgba(34,197,94,.14); }
.panel-title { font-size: 1.25rem; font-weight: 700; color: #fff; margin-bottom: .7rem; }

/* file uploader */
[data-testid="stFileUploaderDropzone"] { background: rgba(6,18,14,.55); border: 2px dashed rgba(74,222,128,.65);
  border-radius: 14px; }
[data-testid="stFileUploaderDropzone"] * { color: #E6F4EA; }
[data-testid="stFileUploaderDropzone"] button { background: #15803D; color: #fff; border: 0; border-radius: 10px; }
[data-testid="stFileUploaderDropzone"] button:hover { background: #16A34A; }
[data-testid="stFileUploaderFile"] * { color: #E6F4EA; }

/* tabs */
.stTabs [data-baseweb="tab"] { color: #CFE8D8; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #4ADE80; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #22C55E; }
.stTabs [data-baseweb="tab-border"] { background-color: rgba(74,222,128,.2); }

/* buttons */
.stButton > button { width: 100%; background: linear-gradient(90deg, #16A34A, #4ADE80); border: 0;
  border-radius: 12px; padding: .6rem 1rem; box-shadow: 0 0 18px rgba(74,222,128,.35); }
.stButton > button p { color: #04210F; font-weight: 700; }
.stButton > button:hover { filter: brightness(1.08); }

/* expander */
[data-testid="stExpander"] { border: 1px solid rgba(74,222,128,.3); border-radius: 12px;
  background: rgba(10,30,22,.6); }
[data-testid="stExpander"] * { color: #E6F4EA; }

/* preview under the uploader */
.prev { display: flex; gap: .9rem; align-items: center; margin-top: .8rem; }
.prev img { width: 120px; height: 90px; object-fit: cover; border-radius: 10px;
            border: 2px solid rgba(74,222,128,.7); box-shadow: 0 0 14px rgba(74,222,128,.4); }
.prev-name { font-weight: 600; word-break: break-all; }
.prev-dim { color: #9DB8A8; font-size: .85rem; }

/* result */
.res-img { width: 100%; border-radius: 14px; border: 3px solid rgba(74,222,128,.85);
           box-shadow: 0 0 22px rgba(74,222,128,.55); display: block; }
.res-label { color: #9DB8A8; font-size: .9rem; }
.res-name { font-size: 2rem; font-weight: 700; margin: .1rem 0 .4rem; line-height: 1.15; }
.pill { display: inline-block; padding: .2rem .8rem; border-radius: 999px; font-size: .82rem; font-weight: 600;
        background: rgba(74,222,128,.16); border: 1px solid rgba(74,222,128,.4); color: #E6F4EA; }
.conf { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-top: .9rem; }
.conf-pct { font-size: 2.4rem; font-weight: 700; line-height: 1.1; }
.ring { width: 78px; height: 78px; border-radius: 50%; flex-shrink: 0; display: flex;
        align-items: center; justify-content: center; }
.ring-in { width: 60px; height: 60px; border-radius: 50%; background: #0B1F18; display: flex;
           align-items: center; justify-content: center; font-size: .8rem; font-weight: 700; color: #E6F4EA; }
.guide { margin-top: .9rem; background: rgba(6,18,14,.55); border: 1px solid rgba(74,222,128,.4);
         border-radius: 14px; padding: .9rem 1.1rem; }
.guide-title { font-weight: 700; color: #4ADE80; margin-bottom: .5rem; }
.guide-item { display: flex; gap: .6rem; margin: .35rem 0; }
.guide-item i { color: #4ADE80; font-style: normal; font-weight: 700; }
.guide-note { color: #9DB8A8; font-size: .78rem; font-style: italic; margin-top: .6rem; }
.empty { text-align: center; padding: 2.2rem 1rem; color: #CFE8D8; }
.empty .big { font-size: 4rem; filter: drop-shadow(0 0 16px rgba(74,222,128,.6)); }
.empty p { margin: .4rem 0 0; }
.sub-title { font-weight: 700; margin: 1rem 0 .5rem; }
.bar-row { margin-bottom: .6rem; }
.bar-head { display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: .2rem; }
.bar-track { background: rgba(255,255,255,.12); border-radius: 999px; height: 10px; overflow: hidden; }
.bar-fill { height: 10px; border-radius: 999px; }

/* ---------- section headings + category cards ---------- */
.sec-head { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: .5rem;
            margin: 1.5rem 0 .8rem; }
.sec-title { font-size: 1.3rem; font-weight: 700; color: #fff; }
.sec-badge { background: rgba(74,222,128,.14); border: 1px solid rgba(74,222,128,.3); border-radius: 10px;
             padding: .3rem .8rem; font-size: .85rem; color: #E6F4EA; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(165px, 1fr)); gap: .8rem; }
.cat { background: #E3F4E8; color: #12261C; border-radius: 14px; padding: .9rem .6rem; text-align: center;
       box-shadow: 0 0 18px rgba(74,222,128,.18); }
.cat-icon { font-size: 2rem; line-height: 1.2; }
.cat-name { font-weight: 600; font-size: .95rem; margin-top: .2rem; }
.cat-type { font-size: .8rem; font-weight: 500; }

/* ---------- bottom sections ---------- */
.bottom { display: grid; grid-template-columns: 1fr 1.25fr 1.1fr; gap: 1rem; margin-top: 1.6rem; }
.box { background: linear-gradient(160deg, rgba(14,48,36,.75), rgba(8,26,20,.85)); border: 1px solid rgba(74,222,128,.3);
       border-radius: 16px; padding: 1.1rem 1.3rem; }
.box-title { font-weight: 700; font-size: 1.1rem; color: #fff; margin-bottom: .7rem; }
.stats { display: grid; gap: .8rem; }
.stat { display: flex; align-items: center; gap: .8rem; }
.stat-icon { font-size: 1.7rem; }
.stat-num { font-size: 1.5rem; font-weight: 700; color: #4ADE80; line-height: 1.1; }
.stat-lbl { font-size: .8rem; color: #9DB8A8; }
.about-line { margin: .25rem 0; font-size: .92rem; }
.about-line b { color: #fff; }
details { background: rgba(6,18,14,.55); border: 1px solid rgba(74,222,128,.22); border-radius: 10px;
          padding: .6rem .9rem; margin-bottom: .5rem; }
details summary { cursor: pointer; font-weight: 600; font-size: .92rem; }
details p { margin: .5rem 0 0; font-size: .88rem; color: #CFE8D8; }

/* ---------- footer ---------- */
.foot { margin-top: 1.6rem; padding: 1rem .4rem; border-top: 1px solid rgba(74,222,128,.2); display: flex;
        flex-wrap: wrap; gap: .6rem 1.5rem; align-items: center; justify-content: space-between;
        font-size: .85rem; color: #CFE8D8; }
.foot b { color: #fff; }

/* ---------- responsive ---------- */
@media (max-width: 1000px) {
  .nav-links, .hero-globe, .hero-slogan { display: none; }
  .bottom { grid-template-columns: 1fr; }
  [data-testid="stHorizontalBlock"]:has(.st-key-upload_panel) { flex-direction: column !important; }
  [data-testid="stHorizontalBlock"]:has(.st-key-upload_panel) > [data-testid="stColumn"],
  [data-testid="stHorizontalBlock"]:has(.st-key-upload_panel) > [data-testid="column"] {
    width: 100% !important; min-width: 100% !important; flex: 1 1 100% !important; }
}
@media (max-width: 640px) {
  .nav { padding: 0 1rem; }
  .nav-tag { display: none; }
  .block-container { padding: 5rem .7rem 1.5rem !important; }
  .hero { padding: 1.3rem 1.1rem; }
  .res-name { font-size: 1.6rem; }
  .grid { grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); }
}
</style>
"""


# --------------------------------------------------------------------------------------
# HTML helpers
# --------------------------------------------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value))


def navbar_html() -> str:
    return (
        '<div class="nav">'
        f'<div class="nav-brand">♻️ <div>Waste<span>Wise</span></div></div>'
        '<div class="nav-links">'
        '<a class="active" href="#home">🏠 Home</a>'
        '<a href="#classifier">🔍 Classifier</a>'
        '<a href="#waste-guide">🍃 Waste Guide</a>'
        '<a href="#about">ℹ️ About Project</a>'
        '<a href="#faq">❓ FAQ</a>'
        "</div>"
        f'<div class="nav-tag">Group {esc(GROUP_ID)}</div>'
        "</div>"
    )


def hero_html() -> str:
    return (
        '<div class="hero anchor" id="home">'
        '<div class="hero-slogan">Reduce<br>Reuse<br>Recycle</div>'
        '<div class="hero-globe">♻️</div>'
        '<div class="hero-chip">🎓 AI-Powered Recycling Assistant</div>'
        '<div class="hero-title">Waste Classification for <b>Recycling.</b></div>'
        '<div class="hero-sub">Identify waste. Discover how to recycle. Make every item count.</div>'
        '<div class="hero-badges">'
        f'<span class="hero-badge">👥 Group: {esc(GROUP_ID)}</span>'
        f'<span class="hero-badge">📁 Project: {esc(PROJECT_ID)}</span>'
        "</div></div>"
    )


def image_to_data_uri(image, max_side: int = 640):
    """Return a base64 JPEG data URI for a PIL image, or None if it cannot be converted."""
    try:
        img = image.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def category_grid_html(items: list) -> str:
    """items: list of (icon, display_name, category, color)."""
    cells = "".join(
        f'<div class="cat"><div class="cat-icon">{esc(icon)}</div>'
        f'<div class="cat-name">{esc(name)}</div>'
        f'<div class="cat-type" style="color:{color};">({esc(category)})</div></div>'
        for icon, name, category, color in items
    )
    return f'<div class="grid">{cells}</div>'


def result_text_html(icon: str, name: str, category: str, confidence: float, color: str) -> str:
    conf_color = confidence_color(confidence, CONFIDENCE_THRESHOLD)
    pct = min(max(confidence, 0.0), 1.0) * 100
    return (
        '<div class="res-label">Predicted Category</div>'
        f'<div class="res-name" style="color:{color};">{esc(icon)} {esc(name)}</div>'
        f'<span class="pill">{esc(category)}</span>'
        '<div class="conf"><div><div class="res-label">Model Confidence</div>'
        f'<div class="conf-pct" style="color:{conf_color};">{pct:.1f}%</div></div>'
        f'<div class="ring" style="background:conic-gradient({conf_color} 0 {pct:.1f}%, '
        f'rgba(255,255,255,0.14) {pct:.1f}% 100%); box-shadow:0 0 16px {conf_color};">'
        f'<div class="ring-in">{pct:.0f}%</div></div></div>'
    )


def guidance_html(bin_text: str, tip: str) -> str:
    return (
        '<div class="guide"><div class="guide-title">🍃 Recycling Guidance</div>'
        f'<div class="guide-item"><i>✔</i><span>{esc(bin_text)}</span></div>'
        f'<div class="guide-item"><i>✔</i><span>{esc(tip)}</span></div>'
        f'<div class="guide-note">ⓘ {esc(DISCLAIMER)}</div></div>'
    )


def top_predictions_html(items: list, bar_color: str) -> str:
    """items: list of (label_text, confidence between 0 and 1)."""
    rows = ""
    for label, conf in items:
        pct = min(max(conf, 0.0), 1.0) * 100
        rows += (
            f'<div class="bar-row"><div class="bar-head"><span>{esc(label)}</span>'
            f"<span>{pct:.1f}%</span></div>"
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{pct:.1f}%; background:{bar_color};"></div></div></div>'
        )
    return rows


# --------------------------------------------------------------------------------------
# App logic
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the Teachable Machine model...")
def get_model():
    return load_model_and_classes()


def load_metrics() -> dict | None:
    path = config.REPORTS_DIR / "metrics.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def percent(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def get_class_names():
    """Real class names from the model, or None if the model is unavailable."""
    try:
        _, class_names = get_model()
        return list(class_names)
    except (ModelNotFoundError, ModelLoadError):
        return None


def set_source(src: str) -> None:
    st.session_state["active_src"] = src


def pick_source(uploaded, photo):
    """Use whichever input (upload or camera) the person touched most recently."""
    active = st.session_state.get("active_src")
    if active == "camera" and photo is not None:
        return photo
    if active == "upload" and uploaded is not None:
        return uploaded
    if uploaded is not None:
        return uploaded
    return photo


def analyse(file_obj):
    """Run the real prediction pipeline. Returns (image, result) or (None, None) after showing an error."""
    size = getattr(file_obj, "size", None)
    if size is not None and size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"The file is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller image.")
        return None, None
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    try:
        image = load_image(file_obj)
        get_model()
        with st.spinner("Analysing the image..."):
            result = predict_image(image, top_k=3)
    except InvalidImageError:
        st.error("This file could not be read as an image. Please upload a valid JPG or PNG.")
    except ModelNotFoundError as exc:
        st.error(f"The Teachable Machine model files are missing. Details: {exc}")
    except ModelLoadError as exc:
        st.error(f"The Teachable Machine model could not be loaded. Details: {exc}")
    except Exception:
        logger.exception("Unexpected error during prediction")
        st.error("Something went wrong while analysing the image. Please try another image.")
    else:
        return image, result
    return None, None


def count_classification(file_obj) -> None:
    """Count each distinct image once per session (not once per rerun)."""
    sig = (
        getattr(file_obj, "name", None),
        getattr(file_obj, "size", None),
        getattr(file_obj, "file_id", None),
    )
    if st.session_state.get("counted_sig") != sig:
        st.session_state["counted_sig"] = sig
        st.session_state["classified_count"] = st.session_state.get("classified_count", 0) + 1


def render_empty_result() -> None:
    st.markdown(
        '<div class="empty"><div class="big">♻️</div>'
        "<p><b>No image yet</b></p>"
        "<p>Upload or take a photo of one waste item to see its class and how to dispose of it.</p></div>",
        unsafe_allow_html=True,
    )


def render_result(result: dict, image) -> None:
    info = get_info(result["label"])
    confidence = result["confidence"]
    color = category_color(info["category"])

    col_image, col_text = st.columns([4, 6], gap="medium")
    with col_image:
        uri = image_to_data_uri(image)
        if uri:
            st.markdown(f'<img class="res-img" src="{uri}" alt="Your photo">', unsafe_allow_html=True)
        else:
            st.image(image, caption="Your photo", width=320)
    with col_text:
        st.markdown(
            result_text_html(info["icon"], info["display_name"], info["category"], confidence, color),
            unsafe_allow_html=True,
        )

    st.caption("A high confidence does not guarantee a correct answer.")
    if not result["is_confident"]:
        st.warning(
            f"Confidence is below {CONFIDENCE_THRESHOLD:.0%}. The result is uncertain, "
            "so please try a clearer photo."
        )
    if result["label"] == "e-waste":
        st.warning(
            "E-waste was the least reliable class in our own tests. "
            "Please check the result yourself before relying on it."
        )

    st.markdown(guidance_html(info["bin"], info["tip"]), unsafe_allow_html=True)

    st.markdown('<div class="sub-title">Top predictions</div>', unsafe_allow_html=True)
    bar_color = confidence_color(confidence, CONFIDENCE_THRESHOLD)
    items = []
    for entry in result["top_k"]:
        entry_info = get_info(entry["label"])
        items.append((f"{entry_info['icon']} {entry_info['display_name']}", entry["confidence"]))
    st.markdown(top_predictions_html(items, bar_color), unsafe_allow_html=True)

    all_df = (
        pd.Series(result["probabilities"], name="probability (%)")
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
        .to_frame()
    )
    st.markdown('<div class="sub-title">Class probability graph</div>', unsafe_allow_html=True)
    st.bar_chart(all_df, color="#4ADE80", height=280)

    with st.expander("All class probabilities (table)"):
        st.dataframe(all_df)


def render_classifier() -> None:
    """Upload + result panels, side by side on desktop."""
    st.markdown('<div class="anchor" id="classifier"></div>', unsafe_allow_html=True)
    n = st.session_state.setdefault("reset_n", 0)

    left, right = st.columns(2, gap="medium")

    # ---- left: upload panel ----
    with left:
        with st.container(key="upload_panel"):
            st.markdown('<div class="panel-title">⬆️ Upload Your Waste Image</div>', unsafe_allow_html=True)
            tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Use camera"])
            with tab_upload:
                uploaded = st.file_uploader(
                    "Choose an image",
                    type=["jpg", "jpeg", "png", "bmp"],
                    key=f"uploader_{n}",
                    on_change=set_source,
                    args=("upload",),
                    help=f"JPG, JPEG, PNG or BMP, up to {MAX_UPLOAD_MB} MB. One item on a plain, light background works best.",
                )
            with tab_camera:
                photo = st.camera_input(
                    "Take a photo",
                    key=f"camera_{n}",
                    on_change=set_source,
                    args=("camera",),
                )
            preview_slot = st.empty()
            st.caption(f"Supported formats: JPG, JPEG, PNG, BMP · up to {MAX_UPLOAD_MB} MB")

    file_obj = pick_source(uploaded, photo)

    # ---- right: result panel ----
    with right:
        with st.container(key="result_panel"):
            st.markdown('<div class="panel-title">✅ Classification Result</div>', unsafe_allow_html=True)
            if file_obj is None:
                render_empty_result()
            else:
                image, result = analyse(file_obj)
                if result is not None:
                    count_classification(file_obj)
                    render_result(result, image)

                    name = getattr(file_obj, "name", None) or "camera photo"
                    dims = ""
                    if hasattr(image, "size"):
                        try:
                            dims = f"{image.size[0]} × {image.size[1]}"
                        except Exception:
                            dims = ""
                    uri = image_to_data_uri(image, max_side=240)
                    if uri:
                        preview_slot.markdown(
                            f'<div class="prev"><img src="{uri}" alt="">'
                            f'<div><div class="prev-name">{esc(name)}</div>'
                            f'<div class="prev-dim">{esc(dims)}</div></div></div>',
                            unsafe_allow_html=True,
                        )
                    if st.button("🔄 Classify Another Image", key="reset_btn"):
                        st.session_state["reset_n"] = n + 1
                        st.session_state["active_src"] = None
                        st.rerun()


def render_categories(class_names) -> None:
    st.markdown('<div class="anchor" id="waste-guide"></div>', unsafe_allow_html=True)
    if class_names is None:
        st.markdown(
            '<div class="sec-head"><div class="sec-title">🧩 Explore Waste Categories</div></div>',
            unsafe_allow_html=True,
        )
        st.error("The Teachable Machine model is not available, so predictions are switched off.")
        return
    items = []
    for name in class_names:
        info = get_info(name)
        items.append((info["icon"], info["display_name"], info["category"], category_color(info["category"])))
    st.markdown(
        '<div class="sec-head"><div class="sec-title">🧩 Explore Waste Categories</div>'
        f'<div class="sec-badge">✔ {len(class_names)} Categories Supported</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(category_grid_html(items), unsafe_allow_html=True)


def render_bottom(class_names) -> None:
    metrics = load_metrics() or {}
    classified = st.session_state.get("classified_count", 0)
    n_classes = len(class_names) if class_names else "n/a"
    names = (
        ", ".join(get_info(c)["display_name"] for c in class_names) if class_names else "not available right now"
    )

    stats = (
        '<div class="box"><div class="box-title">📈 Our Impact</div><div class="stats">'
        f'<div class="stat"><div class="stat-icon">🍃</div><div><div class="stat-num">{classified}</div>'
        '<div class="stat-lbl">Images classified this session</div></div></div>'
        f'<div class="stat"><div class="stat-icon">♻️</div><div><div class="stat-num">{esc(n_classes)}</div>'
        '<div class="stat-lbl">Waste categories</div></div></div>'
        f'<div class="stat"><div class="stat-icon">🎯</div><div><div class="stat-num">'
        f'{esc(percent(metrics.get("accuracy_dataset_test")))}</div>'
        '<div class="stat-lbl">Accuracy on dataset test images</div></div></div>'
        "</div></div>"
    )

    about = (
        '<div class="box anchor" id="about"><div class="box-title">ℹ️ About This Project</div>'
        f'<div class="about-line"><b>{esc(APP_TITLE)}</b></div>'
        f'<div class="about-line"><b>Project ID:</b> {esc(PROJECT_ID)} &nbsp;|&nbsp; <b>Group ID:</b> {esc(GROUP_ID)}</div>'
        f'<div class="about-line"><b>Class:</b> {esc(CLASS_LINE)}</div>'
        f'<div class="about-line"><b>Mentor:</b> {esc(MENTOR)}</div>'
        '<div class="about-line"><b>Technology:</b> Python, Streamlit, Google Teachable Machine</div>'
        "</div>"
    )

    acc_parts = []
    if metrics.get("accuracy_dataset_test") is not None:
        acc_parts.append(
            f"{percent(metrics['accuracy_dataset_test'])} on {metrics.get('dataset_test_images', '?')} "
            "dataset test images the model never saw during training"
        )
    if metrics.get("accuracy_extra_test") is not None:
        acc_parts.append(
            f"{percent(metrics['accuracy_extra_test'])} on {metrics.get('extra_test_images', '?')} "
            "own phone photos (e-waste)"
        )
    if metrics.get("macro_f1") is not None:
        acc_parts.append(f"macro F1 {metrics['macro_f1']:.3f}")
    accuracy_text = (
        "Measured: " + "; ".join(acc_parts) + "." if acc_parts else "Measured accuracy is not available yet."
    )

    faq = (
        '<div class="box anchor" id="faq"><div class="box-title">❓ FAQs</div>'
        "<details><summary>How does the AI model work?</summary>"
        "<p>Your photo is resized and passed to an image classification model trained with Google "
        "Teachable Machine. The model gives a probability for every class and the highest one is shown "
        "as the prediction.</p></details>"
        "<details><summary>Which waste categories are supported?</summary>"
        f"<p>{esc(names)}.</p></details>"
        "<details><summary>How accurate is it?</summary>"
        f"<p>{esc(accuracy_text)}</p></details>"
        "<details><summary>What are the limitations?</summary>"
        "<p>The model only knows its listed classes, so any other object is forced into one of them. "
        "Real-world photos can be harder than training images, and items on brown or cardboard-like "
        "surfaces were often misclassified in our own tests. Use a clear photo of a single item on a "
        "plain background.</p></details>"
        "</div>"
    )
    st.markdown(f'<div class="bottom">{stats}{about}{faq}</div>', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        '<div class="foot">'
        f"<div>♻️ <b>{esc(BRAND)}</b> &nbsp;|&nbsp; B.Tech Deep Learning Project – {esc(UNIVERSITY)}</div>"
        "<div><i>Let's keep our planet clean and green 🌱</i></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(navbar_html(), unsafe_allow_html=True)
    st.markdown(hero_html(), unsafe_allow_html=True)

    render_classifier()

    class_names = get_class_names()
    render_categories(class_names)
    render_bottom(class_names)
    render_footer()


main()