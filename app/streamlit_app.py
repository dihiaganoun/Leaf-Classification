"""Streamlit web app: upload a leaf photo and classify its margin."""

import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent

try:
    import leaf_classifier  # noqa: F401
except ModuleNotFoundError:
    # Package not pip-installed (e.g. on Streamlit Community Cloud): use the source tree
    sys.path.insert(0, str(APP_DIR.parent / "src"))

from leaf_classifier import CLASS_NAMES, load_model, predict
from specimen import MARGINS, accession_number, determination_label, margin_drawing

HIGH_CONFIDENCE = 0.9
LOW_CONFIDENCE = 0.7
GITHUB_URL = "https://github.com/dihiaganoun/Leaf-Classification"
MODEL_URL = "https://huggingface.co/dganoun/leaf-classifier-resnext101"

st.set_page_config(
    page_title="Dentée ou lisse ? · Leaf Classifier",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def render_html(html: str) -> None:
    """Render HTML on one line: Markdown would turn indented lines into code blocks."""
    st.markdown(" ".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Chargement du modèle…")
def get_model():
    return load_model()


render_html(f"<style>{(APP_DIR / 'style.css').read_text(encoding='utf-8')}</style>")

try:
    model = get_model()
except (FileNotFoundError, RuntimeError, OSError) as err:
    st.error(f"Le modèle n'a pas pu être chargé : {err}")
    st.stop()

render_html(
    """
    <div class="eyebrow">Herbier numérique · Détermination du bord</div>
    <h1 class="headline">Dentée <em>ou</em> lisse&nbsp;?</h1>
    <p class="lede">Déposez la photo d'une feuille. Le modèle examine son bord
    et rend sa détermination sur une étiquette d'herbier.</p>
    """
)

uploaded_file = st.file_uploader(
    "Photo de la feuille",
    type=["jpg", "jpeg", "png"],
    help="JPG ou PNG. Une seule feuille, bien centrée, sur fond uni.",
)

if uploaded_file is None:
    render_html(
        '<div class="key">'
        + "".join(
            f"""
            <div class="key-item">
                {margin_drawing(name)}
                <div>
                    <div class="key-latin">{MARGINS[name]['latin']}</div>
                    <div class="key-common">Feuille {MARGINS[name]['label'].lower()}.<br>{MARGINS[name]['desc']}</div>
                </div>
            </div>
            """
            for name in CLASS_NAMES
        )
        + "</div>"
    )
else:
    data = uploaded_file.getvalue()
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Examen du bord de la feuille…"):
        start = time.perf_counter()
        probs = predict(model, image)
        elapsed = time.perf_counter() - start

    col_specimen, col_label = st.columns([1, 1.2], gap="large")
    with col_specimen:
        st.image(image, width="stretch")
        render_html(
            f'<div class="specimen-caption">Spécimen · {uploaded_file.name} · '
            f"{image.width}×{image.height} px</div>"
        )
    with col_label:
        render_html(
            determination_label(
                probs, accession_number(data), elapsed, HIGH_CONFIDENCE, LOW_CONFIDENCE
            )
        )

render_html(
    f"""
    <div class="colophon">
        <span>ResNeXt-101 affiné · 98,68 % de précision sur 984 feuilles de test</span>
        <span><a href="{GITHUB_URL}" target="_blank">Code</a> ·
        <a href="{MODEL_URL}" target="_blank">Poids du modèle</a></span>
    </div>
    """
)
