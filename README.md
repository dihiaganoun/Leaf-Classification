# 🍃 Leaf Classification

Classifies a leaf photo by the shape of its margin: **dentée** (toothed edge) or **lisse** (smooth edge).

The model is a **ResNeXt-101 (32×8d)** fine-tuned with PyTorch on the `feuilles_plantes` dataset (about 6,500 images), served through a **Streamlit** web app.

## Features

- Upload a JPG/PNG leaf photo and get the predicted class
- Probability bar for each class and a confidence level (high / medium / low)
- Warning when the model is unsure
- Inference time display, runs on CPU or GPU

## Project structure

| File | Description |
|------|-------------|
| `streamlit_app.py` | Streamlit web app (inference + UI) |
| `training_notebook.ipynb` | Notebook used to train the model |
| `requirements.txt` | Python dependencies |

## Installation

```bash
pip install -r requirements.txt
```

### Model weights

The trained weights file `resnext101_leaves.pth` (~350 MB) is **not included** in this repository because it exceeds GitHub's file size limit. Place it in the project root, next to `streamlit_app.py`.

## Run the app

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Classes

| Class | Description |
|-------|-------------|
| 🌿 Dentée | Leaf edge cut into small teeth |
| 🍃 Lisse | Continuous leaf edge, no teeth |

## Tips

For the best results, use a single leaf, centered, on a plain and well-lit background.
