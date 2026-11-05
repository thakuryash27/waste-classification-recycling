"""Compare preprocessing variants against the Teachable Machine web preview (read-only)."""

import random

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from src import config
from src.predict import load_image, load_model_and_classes

BATTERY_DIR = config.PROCESSED_DIR / "test" / "battery"
TM_WEB = {
    "battery729.jpg": "e-waste 99%, trash 1%",
    "battery619.jpg": "paper 99%, e-waste 1%",
    "battery60.jpg": "plastic 87%, trash 13%",
    "battery753.jpg": "plastic 78%, e-waste 15%, paper 7%",
}
SIZE = config.IMAGE_SIZE


def center_square(img):
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def tf_resize(img):
    arr = np.asarray(img, dtype="float32")
    return tf.image.resize(arr, SIZE, method="bilinear", antialias=False).numpy()


def norm_a(arr):
    return arr / 127.5 - 1.0


def norm_b(arr):
    return (arr - 127.0) / 127.0


def pil_fit(img, method):
    return np.asarray(ImageOps.fit(img, SIZE, method), dtype="float32")


VARIANTS = {
    "A current: PIL crop + LANCZOS, x/127.5-1": lambda im: norm_a(pil_fit(im, Image.Resampling.LANCZOS)),
    "B crop + PIL BILINEAR, x/127.5-1": lambda im: norm_a(pil_fit(im, Image.Resampling.BILINEAR)),
    "C crop + TF bilinear, no antialias, x/127.5-1": lambda im: norm_a(tf_resize(center_square(im))),
    "D crop + TF bilinear, no antialias, (x-127)/127": lambda im: norm_b(tf_resize(center_square(im))),
    "E squash + TF bilinear, no antialias, x/127.5-1": lambda im: norm_a(tf_resize(im)),
}


def predict_batch(model, arrays):
    return model(np.stack(arrays), training=False).numpy()


def top3_text(probs, class_names):
    order = np.argsort(probs)[::-1][:3]
    return ", ".join(f"{class_names[i]} {probs[i] * 100:.0f}%" for i in order)


def main() -> None:
    model, class_names = load_model_and_classes()

    print("PART 1: four battery images, TM web preview vs each variant\n")
    for name, web in TM_WEB.items():
        path = BATTERY_DIR / name
        if not path.exists():
            print(f"missing: {path}\n")
            continue
        image = load_image(path)
        print(f"{name}\n  {'TM web preview':50s} {web}")
        for label, fn in VARIANTS.items():
            probs = predict_batch(model, [fn(image)])[0]
            print(f"  {label:50s} {top3_text(probs, class_names)}")
        print()

    print("PART 2: accuracy of each variant on validation images (25 per class)\n")
    rng = random.Random(config.SEED)
    images, labels = [], []
    for class_dir in sorted(p for p in (config.PROCESSED_DIR / "val").iterdir() if p.is_dir()):
        if class_dir.name not in class_names:
            continue
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in config.IMAGE_EXTENSIONS)
        for f in rng.sample(files, min(25, len(files))):
            images.append(load_image(f))
            labels.append(class_names.index(class_dir.name))
    y_true = np.array(labels)

    for label, fn in VARIANTS.items():
        preds = []
        for start in range(0, len(images), 50):
            batch = [fn(im) for im in images[start : start + 50]]
            preds.append(predict_batch(model, batch).argmax(axis=1))
        accuracy = float((np.concatenate(preds) == y_true).mean())
        print(f"  {label:50s} accuracy {accuracy:.4f} ({len(y_true)} val images)")


if __name__ == "__main__":
    main()