"""Single-image prediction using ONLY the Google Teachable Machine model.

There is no fallback: if the model cannot be loaded, the error is raised to the caller.
"""

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.tm_model import (
    TM_INPUT_SIZE,
    ModelLoadError,
    ModelNotFoundError,
    load_tm_model,
)

CONFIDENCE_THRESHOLD = 0.60

__all__ = [
    "CONFIDENCE_THRESHOLD",
    "InvalidImageError",
    "ModelLoadError",
    "ModelNotFoundError",
    "load_image",
    "load_model_and_classes",
    "predict_image",
    "preprocess_image",
]


class InvalidImageError(ValueError):
    """Raised when the input cannot be read as an image."""


def load_model_and_classes():
    """The single model entry point of the whole application (no fallback model)."""
    return load_tm_model()


def load_image(source) -> Image.Image:
    """Open a path, bytes, file-like object or PIL image and return an RGB image."""
    try:
        if isinstance(source, Image.Image):
            image = source
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            image = Image.open(path)
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(source))
        else:
            image = Image.open(source)  # file-like object, e.g. a Streamlit upload
        image.load()  # force full decoding so corrupt files fail here
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except FileNotFoundError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        Image.DecompressionBombError,
    ) as exc:
        raise InvalidImageError(f"Could not read the input as a valid image: {exc}") from exc


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Teachable Machine export steps: centre-crop, resize to 224x224, scale pixels to [-1, 1]."""
    fitted = ImageOps.fit(image, TM_INPUT_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(fitted, dtype="float32") / 127.5 - 1.0
    return np.expand_dims(array, axis=0)


def predict_image(source, top_k: int = 3) -> dict:
    """Return the predicted class, confidence and the top-k alternatives."""
    model, class_names = load_model_and_classes()
    image = load_image(source)
    probs = model(preprocess_image(image), training=False).numpy()[0]
    if len(probs) != len(class_names):
        raise RuntimeError("Model output size does not match labels.txt.")

    top_k = max(1, min(top_k, len(class_names)))
    order = np.argsort(probs)[::-1]
    top = [{"label": class_names[i], "confidence": float(probs[i])} for i in order[:top_k]]
    return {
        "label": top[0]["label"],
        "confidence": top[0]["confidence"],
        "is_confident": top[0]["confidence"] >= CONFIDENCE_THRESHOLD,
        "top_k": top,
        "probabilities": {name: float(p) for name, p in zip(class_names, probs)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a waste image.")
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    try:
        result = predict_image(args.image, top_k=args.top_k)
    except (FileNotFoundError, InvalidImageError, ModelLoadError) as exc:
        raise SystemExit(f"Error: {exc}")

    print(f"\nPrediction : {result['label']}")
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
    if not result["is_confident"]:
        print(f"Warning    : confidence is below {CONFIDENCE_THRESHOLD:.0%}, result is uncertain.")
    print("\nTop predictions:")
    for item in result["top_k"]:
        print(f"  {item['label']:12s} {item['confidence'] * 100:6.2f}%")


if __name__ == "__main__":
    main()