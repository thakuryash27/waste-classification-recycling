"""Print the layer structure stored inside the Teachable Machine .h5 file (read-only)."""

import json

import h5py

from src import config

H5_PATH = config.MODELS_DIR / "teachable_machine" / "keras_model.h5"


def describe(node, depth: int = 0) -> None:
    pad = "  " * depth
    if not (isinstance(node, dict) and "class_name" in node):
        return
    cfg = node.get("config", {}) if isinstance(node.get("config"), dict) else {}
    print(f"{pad}- {node['class_name']}  name={cfg.get('name')}")
    for key in ("input_layers", "output_layers"):
        if key in cfg:
            print(f"{pad}    {key} = {str(cfg[key])[:110]}")
    for layer in cfg.get("layers", []) or []:
        describe(layer, depth + 1)


def main() -> None:
    with h5py.File(H5_PATH, "r") as f:
        raw = f.attrs["model_config"]
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        print("Top-level attributes:", list(f.attrs.keys()))
        print("Weight groups       :", list(f["model_weights"].keys())[:20])
    print("\nModel structure:")
    describe(json.loads(text))


if __name__ == "__main__":
    main()