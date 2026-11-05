"""Create a stratified train/val/test split by copying images into data/processed."""

import argparse
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import config

SPLITS = ("train", "val", "test")


def collect_files() -> tuple[list, list]:
    """Return image paths and their class labels from the raw dataset."""
    if not config.RAW_DIR.exists():
        raise SystemExit(f"Raw dataset not found: {config.RAW_DIR}")

    paths, labels = [], []
    for class_dir in sorted(p for p in config.RAW_DIR.iterdir() if p.is_dir()):
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS:
                paths.append(f)
                labels.append(class_dir.name)
    return paths, labels


def copy_split(split: str, paths: list, labels: list) -> None:
    for path, label in tqdm(list(zip(paths, labels)), desc=f"Copying {split}"):
        target_dir = config.PROCESSED_DIR / split / label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_dir / path.name)


def main(force: bool) -> None:
    existing = [config.PROCESSED_DIR / s for s in SPLITS if (config.PROCESSED_DIR / s).exists()]
    if existing and not force:
        raise SystemExit(
            "Processed data already exists. Run with --force to rebuild it."
        )
    for folder in existing:
        shutil.rmtree(folder)

    paths, labels = collect_files()
    print(f"Found {len(paths)} images in {len(set(labels))} classes.")

    holdout = config.VAL_RATIO + config.TEST_RATIO
    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=holdout, stratify=labels, random_state=config.SEED
    )
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p,
        temp_l,
        test_size=config.TEST_RATIO / holdout,
        stratify=temp_l,
        random_state=config.SEED,
    )

    for name, p, l in (("train", train_p, train_l), ("val", val_p, val_l), ("test", test_p, test_l)):
        copy_split(name, p, l)

    # Save a summary table for the report
    rows = []
    for name, l in (("train", train_l), ("val", val_l), ("test", test_l)):
        counts = pd.Series(l).value_counts()
        for cls, n in counts.items():
            rows.append({"split": name, "class": cls, "images": int(n)})
    summary = pd.DataFrame(rows).pivot(index="class", columns="split", values="images")
    summary = summary[list(SPLITS)]
    summary["total"] = summary.sum(axis=1)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.REPORTS_DIR / "split_summary.csv")

    print("\n", summary.to_string())
    print("\nTotals:", summary[list(SPLITS)].sum().to_dict())
    print("Saved: outputs/reports/split_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild the split")
    main(parser.parse_args().force)