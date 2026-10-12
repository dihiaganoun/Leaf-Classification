"""Streamlit web app: upload a leaf photo and classify its margin."""

import time

import streamlit as st
from PIL import Image

from leaf_classifier import CLASS_NAMES, IMAGE_SIZE, load_model, predict
from leaf_classifier.model import get_device

# Display info for each class
CLASS_INFO = {
    "dente": {
        "label": "Dentée",
        "emoji": "🌿",
        "desc": "Bord de la feuille découpé en petites dents (marge dentée).",
        "color": "#e07a2f",
    },
    "lisse": {
        "label": "Lisse",
        "emoji": "🍃",
        "desc": "Bord de la feuille continu, sans dents ni découpes (marge entière).",
        "color": "#2f9e6b",
    },
}
HIGH_CONFIDENCE = 0.9
LOW_CONFIDENCE = 0.7

st.set_page_config(
    page_title="Leaf Classifier",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .hero {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 55%, #52b788 100%);
        border-radius: 18px;
        padding: 2rem 2.2rem;
        color: #ffffff;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(27, 67, 50, 0.25);
    }
    .hero h1 { color: #ffffff; margin: 0 0 .4rem 0; font-size: 2.3rem; }
    .hero p  { color: #d8f3dc; margin: 0; font-size: 1.05rem; }

    .card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        height: 100%;
    }
    .result-label { font-size: .85rem; text-transform: uppercase; letter-spacing: .08em; opacity: .7; }
    .result-class { font-size: 2.4rem; font-weight: 700; line-height: 1.1; margin: .2rem 0 .3rem 0; }
    .result-desc  { opacity: .85; margin-bottom: 1rem; }

    .badge {
        display: inline-block;
        padding: .25rem .75rem;
        border-radius: 999px;
        font-size: .85rem;
        font-weight: 600;
        color: #ffffff;
    }

    .prob-row { margin: .55rem 0; }
    .prob-head { display: flex; justify-content: space-between; font-size: .95rem; margin-bottom: .25rem; }
    .prob-track { background: rgba(128, 128, 128, 0.2); border-radius: 999px; height: 12px; overflow: hidden; }
    .prob-fill  { height: 100%; border-radius: 999px; transition: width .6s ease; }

    [data-testid="stImage"] img { border-radius: 16px; }

    .footer { text-align: center; opacity: .55; font-size: .85rem; margin-top: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Chargement du modèle ResNeXt-101…")
def get_model():
    return load_model()


def render_html(html: str) -> None:
    """Render HTML on one line: Markdown would turn indented lines into code blocks."""
    st.markdown("".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


def confidence_level(confidence: float) -> tuple[str, str]:
    if confidence >= HIGH_CONFIDENCE:
        return "Confiance élevée", "#2f9e6b"
    if confidence >= LOW_CONFIDENCE:
        return "Confiance moyenne", "#d9a21b"
    return "Confiance faible", "#d64545"


def probability_bars(probs: dict[str, float]) -> str:
    bars = ""
    for name in CLASS_NAMES:
        info = CLASS_INFO[name]
        pct = probs[name] * 100
        bars += f"""
        <div class="prob-row">
            <div class="prob-head">
                <span>{info['emoji']} {info['label']}</span>
                <span><b>{pct:.2f}%</b></span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{pct:.2f}%; background:{info['color']};"></div>
            </div>
        </div>
        """
    return bars


try:
    model = get_model()
except (FileNotFoundError, RuntimeError) as err:
    st.error(f"Impossible de charger le modèle : {err}")
    st.stop()

# === Sidebar ===
with st.sidebar:
    st.markdown("## 🍃 Leaf Classifier")
    st.caption("Classification de la marge des feuilles")
    st.divider()

    st.markdown("### Classes")
    for name in CLASS_NAMES:
        info = CLASS_INFO[name]
        st.markdown(f"**{info['emoji']} {info['label']}**  \n{info['desc']}")

    st.divider()
    st.markdown("### Modèle")
    st.markdown(
        f"""
        - **Architecture :** ResNeXt-101 (32×8d)
        - **Entrée :** {IMAGE_SIZE}×{IMAGE_SIZE} RGB
        - **Classes :** {len(CLASS_NAMES)}
        - **Device :** `{get_device().type.upper()}`
        """
    )

    st.divider()
    st.markdown("### Conseils")
    st.caption(
        "Pour de meilleurs résultats : une seule feuille, bien centrée, "
        "sur un fond uni et bien éclairée."
    )

# === Header ===
st.markdown(
    """
    <div class="hero">
        <h1>🍃 Leaf Classifier</h1>
        <p>Importez la photo d'une feuille : le modèle prédit si sa marge est <b>dentée</b> ou <b>lisse</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choisissez une image de feuille",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptés : JPG, JPEG, PNG",
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyse de la feuille…"):
        start = time.perf_counter()
        probs = predict(model, image)
        elapsed_ms = (time.perf_counter() - start) * 1000

    pred_name = max(probs, key=probs.get)
    confidence = probs[pred_name]
    info = CLASS_INFO[pred_name]
    level, level_color = confidence_level(confidence)

    col1, col2 = st.columns([1, 1.15], gap="large")

    with col1:
        st.image(image, caption=uploaded_file.name, width="stretch")
        st.caption(f"Dimensions : {image.width} × {image.height} px")

    with col2:
        render_html(
            f"""
            <div class="card">
                <div class="result-label">Prédiction</div>
                <div class="result-class" style="color:{info['color']};">{info['emoji']} {info['label']}</div>
                <div class="result-desc">{info['desc']}</div>
                <span class="badge" style="background:{level_color};">{level} · {confidence * 100:.2f}%</span>
                <div style="margin-top:1.3rem;" class="result-label">Probabilités par classe</div>
                {probability_bars(probs)}
            </div>
            """
        )

        m1, m2 = st.columns(2)
        m1.metric("Confiance", f"{confidence * 100:.2f}%")
        m2.metric("Temps d'inférence", f"{elapsed_ms:.0f} ms")

        if confidence < LOW_CONFIDENCE:
            st.warning(
                "Le modèle hésite sur cette image. Essayez une photo plus nette, "
                "avec une seule feuille sur un fond uni."
            )
else:
    st.info("⬆️ Importez une image pour lancer la classification.")

st.markdown(
    '<div class="footer">Leaf Classifier · ResNeXt-101 · PyTorch & Streamlit</div>',
    unsafe_allow_html=True,
)
