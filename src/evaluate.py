"""Evaluate the Teachable Machine model on held-out images using the exact app pipeline."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from src import config
from src.predict import (
    CONFIDENCE_THRESHOLD,
    InvalidImageError,
    load_image,
    load_model_and_classes,
    preprocess_image,
)

TEST_ROOTS = [config.PROCESSED_DIR / "test", config.PROJECT_ROOT / "data" / "extra_test"]
BATCH_SIZE = 32


def collect_samples(class_names: list[str]) -> list[tuple]:
    samples = []
    for root in TEST_ROOTS:
        if not root.exists():
            continue
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if class_dir.name not in class_names:
                print(f"Skipping folder that is not in labels.txt: {class_dir}")
                continue
            label = class_names.index(class_dir.name)
            files = sorted(
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS
            )
            samples.extend((f, label) for f in files)
    return samples


def predict_all(model, samples: list[tuple]):
    kept, probs_list, skipped = [], [], []
    for start in tqdm(range(0, len(samples), BATCH_SIZE), desc="Evaluating"):
        arrays, chunk_kept = [], []
        for path, label in samples[start : start + BATCH_SIZE]:
            try:
                arrays.append(preprocess_image(load_image(path)))
            except InvalidImageError:
                skipped.append(path)
                continue
            chunk_kept.append((path, label))
        if arrays:
            probs_list.append(model(np.concatenate(arrays), training=False).numpy())
            kept.extend(chunk_kept)
    return kept, np.concatenate(probs_list), skipped


def plot_confusion_matrix(cm, class_names, normalize: bool) -> None:
    if normalize:
        sums = cm.sum(axis=1, keepdims=True)
        data = np.divide(cm.astype(float), sums, out=np.zeros(cm.shape), where=sums > 0)
    else:
        data = cm
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        data, annot=True, fmt=".2f" if normalize else "d", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8},
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion matrix (normalized)" if normalize else "Confusion matrix (counts)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    name = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
    plt.savefig(config.PLOTS_DIR / name, dpi=150)
    plt.close()


def plot_f1_per_class(report: dict, names: list[str]) -> None:
    scores = [report[n]["f1-score"] for n in names]
    order = np.argsort(scores)
    plt.figure(figsize=(9, 5))
    plt.barh([names[i] for i in order], [scores[i] for i in order], color="seagreen")
    plt.xlim(0, 1.0)
    plt.xlabel("F1-score")
    plt.title("Per-class F1-score (Teachable Machine model)")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "f1_per_class.png", dpi=150)
    plt.close()


def main() -> None:
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model, class_names = load_model_and_classes()
    samples = collect_samples(class_names)
    if not samples:
        raise SystemExit("No test images found in data/processed/test or data/extra_test.")

    kept, probs, skipped = predict_all(model, samples)
    y_true = np.array([label for _, label in kept])
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    all_labels = list(range(len(class_names)))
    supported = sorted(set(y_true.tolist()))
    missing = [class_names[i] for i in all_labels if i not in supported]
    names = [class_names[i] for i in supported]

    accuracy = float((y_true == y_pred).mean())
    top3 = float(np.mean([t in np.argsort(p)[-3:] for t, p in zip(y_true, probs)]))
    macro = precision_recall_fscore_support(y_true, y_pred, labels=supported, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=supported, average="weighted", zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=supported, target_names=names, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=supported, target_names=names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    confident = confidence >= CONFIDENCE_THRESHOLD
    correct = y_true == y_pred
    dataset_root = config.PROCESSED_DIR / "test"
    from_dataset = np.array([dataset_root in path.parents for path, _ in kept])
    metrics = {
        "model": "Google Teachable Machine (Keras export)",
        "test_images": int(len(y_true)),
        "skipped_unreadable_images": len(skipped),
        "classes_evaluated": names,
        "classes_without_test_images": missing,
        "accuracy": accuracy,
        "top3_accuracy": top3,
        "dataset_test_images": int(from_dataset.sum()),
        "accuracy_dataset_test": float(correct[from_dataset].mean()) if from_dataset.any() else None,
        "extra_test_images": int((~from_dataset).sum()),
        "accuracy_extra_test": float(correct[~from_dataset].mean()) if (~from_dataset).any() else None,
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "share_confident": float(confident.mean()),
        "accuracy_when_confident": float(correct[confident].mean()) if confident.any() else None,
        "accuracy_when_uncertain": float(correct[~confident].mean()) if (~confident).any() else None,
    }
    with open(config.REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    (config.REPORTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(config.REPORTS_DIR / "confusion_matrix.csv")
    pd.DataFrame(
        {
            "image": [str(p.relative_to(config.PROJECT_ROOT)) for p, _ in kept],
            "true": [class_names[t] for t in y_true],
            "predicted": [class_names[p] for p in y_pred],
            "confidence": confidence.round(4),
        }
    ).to_csv(config.REPORTS_DIR / "predictions.csv", index=False)

    plot_confusion_matrix(cm, class_names, normalize=False)
    plot_confusion_matrix(cm, class_names, normalize=True)
    plot_f1_per_class(report_dict, names)

    print("\n" + report_text)
    print("Test images        :", metrics["test_images"], f"(unreadable skipped: {len(skipped)})")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Top-3 accuracy     : {top3:.4f}")
    if from_dataset.any():
        print(f"Accuracy, dataset test images only : {correct[from_dataset].mean():.4f} ({int(from_dataset.sum())} images)")
    if (~from_dataset).any():
        print(f"Accuracy, extra test (own photos)  : {correct[~from_dataset].mean():.4f} ({int((~from_dataset).sum())} images)")
    print(f"Macro F1           : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1        : {metrics['weighted_f1']:.4f}")
    print(f"Confidence >= {CONFIDENCE_THRESHOLD:.0%}   : {confident.mean() * 100:.1f}% of images")
    if metrics["accuracy_when_confident"] is not None:
        print(f"Accuracy if confident: {metrics['accuracy_when_confident']:.4f}")
    if metrics["accuracy_when_uncertain"] is not None:
        print(f"Accuracy if uncertain: {metrics['accuracy_when_uncertain']:.4f}")
    if missing:
        print("\nNOT evaluated (no test images):", ", ".join(missing))

    pairs = [
        (class_names[i], class_names[j], int(cm[i, j]))
        for i in all_labels for j in all_labels if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    print("\nTop confusions (true -> predicted : count):")
    for true_name, pred_name, count in pairs[:8]:
        print(f"  {true_name} -> {pred_name} : {count}")
    print("\nSaved reports in outputs/reports and plots in outputs/plots")


if __name__ == "__main__":
    main()