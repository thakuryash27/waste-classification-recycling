"""Check that the exported Teachable Machine model loads and runs."""

import numpy as np

from src.tm_model import TM_MODEL_PATH, ModelLoadError, ModelNotFoundError, load_tm_model


def main() -> None:
    try:
        model, labels = load_tm_model()
    except (ModelNotFoundError, ModelLoadError) as exc:
        raise SystemExit(f"FAILED: {exc}")

    print("Model file    :", TM_MODEL_PATH)
    print("Classes found :", len(labels))
    for index, name in enumerate(labels):
        print(f"  {index:2d}  {name}")

    noise = np.random.default_rng(0).integers(0, 255, size=(1, 224, 224, 3)).astype("float32")
    probs = model(noise / 127.5 - 1.0, training=False).numpy()[0]
    print("\nOutput shape  :", probs.shape)
    print("Sum of probs  :", round(float(probs.sum()), 4), "(should be close to 1.0)")
    print("\nLoaded OK. The model can be used for predictions.")


if __name__ == "__main__":
    main()