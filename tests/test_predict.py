"""Tests for the Teachable Machine prediction pipeline."""

import io

import pytest
from PIL import Image

from src import config, tm_model
from src.predict import (
    InvalidImageError,
    ModelNotFoundError,
    load_image,
    load_model_and_classes,
    predict_image,
    preprocess_image,
)
from src.tm_model import ModelLoadError, load_labels

TEST_DIR = config.PROCESSED_DIR / "test"

needs_model = pytest.mark.skipif(
    not (tm_model.TM_MODEL_PATH.exists() and tm_model.TM_LABELS_PATH.exists()),
    reason="Teachable Machine model files not found",
)
needs_data = pytest.mark.skipif(not TEST_DIR.exists(), reason="test split not found")


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_load_image_converts_to_rgb(mode):
    assert load_image(Image.new(mode, (40, 30))).mode == "RGB"


def test_load_image_from_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (50, 50), (200, 10, 10)).save(buffer, format="PNG")
    assert load_image(buffer.getvalue()).size == (50, 50)


def test_preprocess_shape_dtype_and_range():
    batch = preprocess_image(Image.new("RGB", (300, 200), (255, 255, 255)))
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == "float32"
    assert -1.0 <= batch.min() and batch.max() <= 1.0


def test_invalid_bytes_raise_clear_error():
    with pytest.raises(InvalidImageError):
        load_image(b"this is definitely not an image")


def test_missing_image_file_raises():
    with pytest.raises(FileNotFoundError):
        load_image("this_file_does_not_exist.jpg")


def test_labels_parsing(tmp_path):
    good = tmp_path / "labels.txt"
    good.write_text("0 battery\n1 white-glass\n", encoding="utf-8")
    assert load_labels(good) == ["battery", "white-glass"]

    bad = tmp_path / "bad_labels.txt"
    bad.write_text("0 battery\n5 metal\n", encoding="utf-8")
    with pytest.raises(ModelLoadError):
        load_labels(bad)


def test_missing_model_gives_clear_error_and_no_fallback(monkeypatch, tmp_path):
    tm_model.load_tm_model.cache_clear()
    monkeypatch.setattr(tm_model, "TM_MODEL_PATH", tmp_path / "does_not_exist.h5")
    with pytest.raises(ModelNotFoundError):
        predict_image(Image.new("RGB", (50, 50)))
    tm_model.load_tm_model.cache_clear()


@needs_model
def test_prediction_is_a_valid_distribution():
    result = predict_image(Image.new("RGB", (500, 400), (120, 80, 40)), top_k=3)
    _, class_names = load_model_and_classes()
    assert result["label"] in class_names
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_k"]) == 3
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3


@needs_model
@needs_data
def test_sanity_accuracy_on_a_few_test_images():
    """Bug guard only: wrong preprocessing or weights would give near-chance accuracy."""
    _, class_names = load_model_and_classes()
    correct, total = 0, 0
    for class_dir in sorted(p for p in TEST_DIR.iterdir() if p.is_dir()):
        if class_dir.name not in class_names:
            continue
        for image_path in sorted(class_dir.iterdir())[:5]:
            total += 1
            correct += predict_image(image_path)["label"] == class_dir.name
    assert total > 0
    assert correct / total >= 0.70