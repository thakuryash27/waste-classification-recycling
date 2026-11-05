"""Create a balanced subset of TRAINING images to upload to Google Teachable Machine."""

import argparse
import random
import shutil

from src import config

OUTPUT_DIR = config.PROJECT_ROOT / "data" / "teachable_machine" / "upload"


def main(per_class: int, force: bool) -> None:
    train_dir = config.PROCESSED_DIR / "train"
    if not train_dir.exists():
        raise SystemExit("data/processed/train not found. Run: python -m src.prepare_data")

    class_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    existing = [OUTPUT_DIR / p.name for p in class_dirs if (OUTPUT_DIR / p.name).exists()]
    if existing and not force:
        raise SystemExit("Output folders already exist. Use --force to rebuild them.")
    for folder in existing:
        shutil.rmtree(folder)

    rng = random.Random(config.SEED)
    total = 0
    for class_dir in class_dirs:
        images = sorted(
            f for f in class_dir.iterdir() if f.suffix.lower() in config.IMAGE_EXTENSIONS
        )
        chosen = rng.sample(images, min(per_class, len(images)))
        target = OUTPUT_DIR / class_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for image in chosen:
            shutil.copy2(image, target / image.name)
        print(f"{class_dir.name:12s} {len(chosen)} images")
        total += len(chosen)
    print(f"\nTotal: {total} images in {OUTPUT_DIR}")
    print("Other folders inside it (for example e-waste) were not touched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.per_class, args.force)