"""Central configuration shared by the app, the evaluation script and the tests."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Class order must match the training ImageFolder order (alphabetical folder names).
CLASS_NAMES = ["dente", "lisse"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Override with the LEAF_MODEL_PATH environment variable if the weights live elsewhere.
DEFAULT_MODEL_PATH = Path(
    os.environ.get("LEAF_MODEL_PATH", PROJECT_ROOT / "models" / "resnext101_leaves.pth")
)
