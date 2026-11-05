"""Scan the raw dataset: count images per class and detect corrupt files."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("outputs/reports")
PLOT_DIR = Path("outputs/plots")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_class_folders(root: Path) -> list[Path]:
    """Return folders that directly contain image files (one per class)."""
    folders = []
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        has_images = any(
            f.suffix.lower() in IMAGE_EXTENSIONS for f in folder.iterdir() if f.is_file()
        )
        if has_images:
            folders.append(folder)
    return folders


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main() -> None:
    if not RAW_DIR.exists():
        raise SystemExit(f"Folder not found: {RAW_DIR.resolve()}")

    class_folders = find_class_folders(RAW_DIR)
    if not class_folders:
        raise SystemExit("No image folders found inside data/raw.")

    rows = []
    corrupt_files = []
    for folder in class_folders:
        images = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        bad = [f for f in images if not is_valid_image(f)]
        corrupt_files.extend(bad)
        rows.append({"class": folder.name, "images": len(images), "corrupt": len(bad)})

    df = pd.DataFrame(rows).sort_values("images", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))
    print("\nNumber of classes :", len(df))
    print("Total images      :", df["images"].sum())
    print("Corrupt images    :", len(corrupt_files))
    print("Largest/smallest  :", round(df["images"].max() / df["images"].min(), 2))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "class_distribution.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(df["class"], df["images"], color="seagreen")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of images")
    plt.title("Class distribution of the raw dataset")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "class_distribution.png", dpi=150)
    print("\nSaved: outputs/reports/class_distribution.csv")
    print("Saved: outputs/plots/class_distribution.png")


if __name__ == "__main__":
    main()