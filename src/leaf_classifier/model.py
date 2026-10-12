"""Model construction, weight loading and inference."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from leaf_classifier.config import (
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
)

_LFS_POINTER_PREFIX = b"version https://git-lfs"

# Same preprocessing as the evaluation transform used during training.
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNeXt-101 32x8d with a new classification head (no pretrained download)."""
    model = models.resnext101_32x8d(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _check_weights_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at '{path}'. "
            "Run 'git lfs pull' or set the LEAF_MODEL_PATH environment variable."
        )
    with path.open("rb") as f:
        if f.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
            raise RuntimeError(
                f"'{path}' is a Git LFS pointer, not the real weights. "
                "Install Git LFS and run 'git lfs pull'."
            )


def load_model(path: str | Path | None = None, device: torch.device | None = None) -> nn.Module:
    """Load the fine-tuned weights and return the model in eval mode."""
    path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    device = device or get_device()
    _check_weights_file(path)

    model = build_model()
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def preprocess(image: Image.Image) -> torch.Tensor:
    """PIL image -> normalized tensor of shape (1, 3, IMAGE_SIZE, IMAGE_SIZE)."""
    return TRANSFORM(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict(model: nn.Module, image: Image.Image) -> dict[str, float]:
    """Return the probability of each class, keyed by class name."""
    device = next(model.parameters()).device
    logits = model(preprocess(image).to(device))
    probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    return dict(zip(CLASS_NAMES, probs))
