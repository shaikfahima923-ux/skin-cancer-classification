"""Class counts and class weights for the nine-class skin lesion dataset.

Weights are computed from the TRAINING subset only (the same 80% split that
train_baseline.py uses: validation_split=0.20, seed=42).

Run on its own:  python src/class_weights.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    ROOT / "dataset" / "raw"
    / "Skin cancer ISIC The International Skin Imaging Collaboration"
)
TRAIN_DIR = DATA_ROOT / "Train"
TEST_DIR = DATA_ROOT / "Test"

RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
VAL_SPLIT = 0.20


def get_train_file_paths():
    """Return (class_names, file paths of the 80% training subset)."""
    ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
    )
    return ds.class_names, list(ds.file_paths)


def count_by_class(file_paths, class_names):
    """Count images per class (class = parent folder name)."""
    counts = {name: 0 for name in class_names}
    for path in file_paths:
        counts[Path(path).parent.name] += 1
    return counts


def compute_class_weights(counts, class_names, mode="balanced"):
    """Return {class_index: weight}.

    balanced: n_samples / (n_classes * count)
    sqrt:     square root of the balanced weight (gentler)
    """
    totals = np.array([counts[name] for name in class_names], dtype=float)
    balanced = totals.sum() / (len(class_names) * totals)

    if mode == "balanced":
        weights = balanced
    elif mode == "sqrt":
        weights = np.sqrt(balanced)
    else:
        raise ValueError("mode must be 'balanced' or 'sqrt'")

    return {i: float(w) for i, w in enumerate(weights)}


def save_class_info(counts, weights_by_mode, class_names):
    """Write imbalance_class_counts.csv and imbalance_class_weights.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {"class_index": range(len(class_names)),
         "class_name": class_names,
         "train_images": [counts[n] for n in class_names]}
    )
    df["share_percent"] = (
        100 * df["train_images"] / df["train_images"].sum()
    ).round(2)
    for mode, weights in weights_by_mode.items():
        df[f"weight_{mode}"] = [round(weights[i], 4)
                                for i in range(len(class_names))]
    df.to_csv(RESULTS_DIR / "imbalance_class_counts.csv", index=False)

    with open(RESULTS_DIR / "imbalance_class_weights.json", "w",
              encoding="utf-8") as f:
        json.dump(
            {mode: {class_names[i]: w for i, w in weights.items()}
             for mode, weights in weights_by_mode.items()},
            f, indent=4,
        )
    return df


if __name__ == "__main__":
    names, paths = get_train_file_paths()
    counts = count_by_class(paths, names)
    weights = {m: compute_class_weights(counts, names, m)
               for m in ("balanced", "sqrt")}
    table = save_class_info(counts, weights, names)
    print("\nTraining-subset class distribution and weights:\n")
    print(table.to_string(index=False))
    print("\nSaved: results/imbalance_class_counts.csv and "
          "results/imbalance_class_weights.json")