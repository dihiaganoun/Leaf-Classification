# 🍃 Leaf Classification

[![CI](https://github.com/dihiaganoun/Leaf-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/dihiaganoun/Leaf-Classification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
[![Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://leaf-margin-classifier.streamlit.app)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Hub-yellow)](https://huggingface.co/dganoun/leaf-classifier-resnext101)

Classifies a leaf photo by the shape of its margin: **dentée** (toothed edge) or **lisse** (smooth edge).

A **ResNeXt-101 (32×8d)** pretrained on ImageNet is fine-tuned on ~6,500 leaf images, then served through a **Streamlit** web app.

**▶ [Try the live demo](https://leaf-margin-classifier.streamlit.app)**, no installation needed.

![App screenshot](docs/screenshot.png)

## Results

Evaluated on the held-out **test split** (15% of the dataset, 984 images, never seen during training):

| Metric | Score |
|--------|-------|
| Accuracy | **98.68%** |
| F1 | 0.9861 |
| Precision | 0.9935 |
| Recall | 0.9787 |
| Balanced accuracy | 0.9864 |
| Cohen's kappa | 0.9735 |

| Class | Precision | Recall | F1 | Images |
|-------|-----------|--------|----|--------|
| dente | 0.99 | 0.98 | 0.99 | 470 |
| lisse | 0.98 | 0.99 | 0.99 | 514 |

These numbers come from `notebooks/training.ipynb` (Colab, GPU).

### Local check of the published weights

`scripts/evaluate.py` re-runs the published weights on a stratified sample of the test split (150 images per class, CPU):

| Sample | Accuracy | Macro F1 | Errors |
|--------|----------|----------|--------|
| 300 images | 99.67% | 0.9967 | 1 (a toothed leaf predicted smooth) |

<img src="docs/confusion_matrix.png" width="420" alt="Confusion matrix">

The `dente` sample is exactly from the held-out test split. The local copy of the dataset has 15 fewer `lisse` images than the one used for training, so the `lisse` split cannot be rebuilt identically and that half of the sample may include training images. Treat this table as a check that the weights work, and the test-split table above as the real score.

## Project structure

```
Leaf-Classification/
├── app/
│   └── streamlit_app.py      # Web interface
├── src/leaf_classifier/
│   ├── config.py             # Classes, image size, model path
│   ├── model.py              # Model building, weight loading, inference
│   └── data.py               # Train/val/test split (same as the notebook)
├── scripts/
│   └── evaluate.py           # Evaluation on a test sample
├── tests/                    # Unit tests (no weights needed)
├── notebooks/
│   └── training.ipynb        # Training and full test evaluation
├── docs/                     # Screenshot, metrics, confusion matrix
├── Dockerfile
├── requirements.txt          # Pinned runtime dependencies
└── requirements-dev.txt      # + evaluation and test tools
```

## Getting started

### 1. Clone

```bash
git clone https://github.com/dihiaganoun/Leaf-Classification.git
cd Leaf-Classification
```

The trained weights (~350 MB) are hosted on [Hugging Face Hub](https://huggingface.co/dganoun/leaf-classifier-resnext101). They are downloaded automatically the first time the model is loaded, then cached.

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` also installs the `leaf_classifier` package in editable mode.

### 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501.

To use a local copy of the weights instead, put it in `models/resnext101_leaves.pth` or set `LEAF_MODEL_PATH=/path/to/weights.pth`.

### Docker

```bash
docker build -t leaf-classifier .
docker run -p 8501:8501 leaf-classifier
```

The image uses CPU-only PyTorch to stay small and downloads the weights at build time. The live demo runs on Streamlit Community Cloud, which uses `app/requirements.txt` (CPU-only PyTorch).

## Using the model in Python

```python
from PIL import Image
from leaf_classifier import load_model, predict

model = load_model()
print(predict(model, Image.open("leaf.jpg")))
# e.g. {'dente': 0.0003, 'lisse': 0.9997}
```

## Development

```bash
pip install -r requirements-dev.txt
pytest                                   # fast, uses a stub model
python scripts/evaluate.py --data-dir path/to/feuilles_plantes --per-class 150
```

The dataset is not versioned. `--data-dir` must contain one folder per class (`dente/`, `lisse/`).

GitHub Actions runs the tests on every push and pull request.

## Model and training

- **Backbone:** ResNeXt-101 32×8d, ImageNet weights
- **Fine-tuning:** `layer3`, `layer4` and the new `fc` head are trained, earlier layers are frozen
- **Split:** stratified 70% train / 15% validation / 15% test (seed 0)
- **Optimizer:** AdamW, lr 1e-4, `ReduceLROnPlateau` on validation loss
- **Training:** 20 epochs, batch size 32, input 224×224, ImageNet normalization

## Known limitations

Found while reviewing the training notebook. They do not affect the published results above, which were measured on the saved weights, but they matter for retraining:

- **Data augmentation is not applied.** The train and test subsets share one dataset object, and its transform is switched to the evaluation transform right after the train loader is built. Training therefore ran without augmentation.
- **"Best model" is actually the last epoch.** `model.state_dict().copy()` is a shallow copy, so the saved "best" state keeps changing with training. The saved weights are from epoch 20.
- **The AUC printed in the notebook (0.0009) is inverted.** The score used is the `lisse` probability while the positive class is `dente`. The real AUC is about 0.999.
- Accuracy on real-world photos (cluttered backgrounds, different lighting) has not been measured.

## License

[MIT](LICENSE)
