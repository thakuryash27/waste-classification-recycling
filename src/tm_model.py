"""Loader for the Google Teachable Machine (Keras export). There is no fallback classifier."""

import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import h5py
import keras
import numpy as np

from src import config

TM_DIR = config.MODELS_DIR / "teachable_machine"
TM_MODEL_PATH = TM_DIR / "keras_model.h5"
TM_LABELS_PATH = TM_DIR / "labels.txt"
TM_INPUT_SIZE = (224, 224)


class ModelNotFoundError(FileNotFoundError):
    """The Teachable Machine model or labels file is missing."""


class ModelLoadError(RuntimeError):
    """The Teachable Machine model exists but could not be loaded or is inconsistent."""


class _CompatDepthwiseConv2D(keras.layers.DepthwiseConv2D):
    """Accept the legacy 'groups' argument stored in Teachable Machine .h5 files."""

    def __init__(self, *args, groups=None, **kwargs):
        super().__init__(*args, **kwargs)


def load_labels(path: Path = TM_LABELS_PATH) -> list[str]:
    """Read labels.txt (format: '<index> <class name>' per line) in index order."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    labels = []
    for position, line in enumerate(lines):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) != position:
            raise ModelLoadError(f"Unexpected labels.txt format at line {position + 1}: {line!r}")
        labels.append(parts[1])
    if not labels:
        raise ModelLoadError("labels.txt is empty.")
    return labels


def _flatten_single_io(node) -> None:
    """Make single-element input/output lists of nested Functional configs flat (Keras 3 style)."""
    if isinstance(node, dict):
        cfg = node.get("config")
        if node.get("class_name") in ("Functional", "Model") and isinstance(cfg, dict):
            for key in ("input_layers", "output_layers"):
                value = cfg.get(key)
                if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
                    cfg[key] = value[0]
        for child in node.values():
            _flatten_single_io(child)
    elif isinstance(node, list):
        for child in node:
            _flatten_single_io(child)


def _write_patched_copy(source: Path, target: Path) -> None:
    """Copy the .h5 file and fix its model_config. The original file is never modified."""
    shutil.copy2(source, target)
    with h5py.File(target, "r+") as f:
        raw = f.attrs.get("model_config")
        if raw is None:
            raise ModelLoadError("The .h5 file has no model_config.")
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        model_config = json.loads(text)
        _flatten_single_io(model_config)
        del f.attrs["model_config"]
        f.attrs["model_config"] = json.dumps(model_config)


@lru_cache(maxsize=1)
def load_tm_model() -> tuple[keras.Model, list[str]]:
    """Load the exported Teachable Machine model and its labels (cached after first success)."""
    for path in (TM_MODEL_PATH, TM_LABELS_PATH):
        if not path.exists():
            raise ModelNotFoundError(f"Teachable Machine file not found: {path}")

    labels = load_labels()
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            patched = Path(tmp) / "keras_model.h5"
            _write_patched_copy(TM_MODEL_PATH, patched)
            model = keras.models.load_model(
                patched,
                compile=False,
                custom_objects={"DepthwiseConv2D": _CompatDepthwiseConv2D},
            )
        probe = model(np.zeros((1, *TM_INPUT_SIZE, 3), dtype="float32"), training=False)
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Could not load the Teachable Machine model: {exc}") from exc

    if probe.shape[-1] != len(labels):
        raise ModelLoadError(
            f"Model outputs {probe.shape[-1]} classes but labels.txt has {len(labels)} entries."
        )
    return model, labels