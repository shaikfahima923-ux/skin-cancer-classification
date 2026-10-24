"""EfficientNetB0 with class-imbalance handling (Member 3A).

Identical to src/train_baseline.py (same split, seed, 224x224, batch size,
augmentation, two-stage training, callbacks) except for the loss:

  --loss weighted : cross-entropy + Keras class_weight (default)
  --loss focal    : focal loss (optionally + class weights via --focal-weights)

Examples
  python src/train_weighted.py
  python src/train_weighted.py --weight-mode sqrt
  python src/train_weighted.py --loss focal --gamma 2.0
"""

import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from class_weights import (
    TRAIN_DIR, TEST_DIR, RESULTS_DIR, MODELS_DIR,
    IMAGE_SIZE, BATCH_SIZE, SEED, VAL_SPLIT,
    count_by_class, compute_class_weights, save_class_info,
)

NUM_CLASSES = 9

parser = argparse.ArgumentParser()
parser.add_argument("--loss", choices=["weighted", "focal"],
                    default="weighted")
parser.add_argument("--weight-mode", choices=["balanced", "sqrt"],
                    default="balanced")
parser.add_argument("--gamma", type=float, default=2.0,
                    help="focal loss gamma")
parser.add_argument("--focal-weights", action="store_true",
                    help="also apply class weights with focal loss")
parser.add_argument("--name", default=None,
                    help="run name used in output files")
args = parser.parse_args()

RUN_NAME = args.name or (
    f"weighted_{args.weight_mode}" if args.loss == "weighted"
    else f"focal_g{args.gamma:g}"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)

# ==================================================
# 1. DATA (same calls and seed as the baseline)
# ==================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=VAL_SPLIT, subset="training", seed=SEED,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int",
    shuffle=True,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=VAL_SPLIT, subset="validation", seed=SEED,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int",
    shuffle=False,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    label_mode="int", shuffle=False,
)

class_names = train_ds.class_names
assert len(class_names) == NUM_CLASSES, "Expected 9 classes"
assert class_names == val_ds.class_names == test_ds.class_names, \
    "Class order differs between splits"

# ==================================================
# 2. CLASS WEIGHTS (training subset labels only)
# ==================================================
counts = count_by_class(train_ds.file_paths, class_names)
weights_by_mode = {
    m: compute_class_weights(counts, class_names, m)
    for m in ("balanced", "sqrt")
}
save_class_info(counts, weights_by_mode, class_names)
class_weight = weights_by_mode[args.weight_mode]

print("\nClass counts (training subset):")
for i, name in enumerate(class_names):
    print(f"{i} {name:35s} n={counts[name]:5d} "
          f"weight={class_weight[i]:.3f}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ==================================================
# 3. LOSS
# ==================================================
def make_focal_loss(gamma):
    """Focal loss for integer labels and softmax outputs."""
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        one_hot = tf.one_hot(y_true, NUM_CLASSES)
        p_t = tf.reduce_sum(one_hot * y_pred, axis=-1)
        return -tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
    return focal_loss


if args.loss == "weighted":
    loss_fn = "sparse_categorical_crossentropy"
    fit_class_weight = class_weight
else:
    loss_fn = make_focal_loss(args.gamma)
    fit_class_weight = class_weight if args.focal_weights else None

# ==================================================
# 4. MODEL (same as baseline)
# ==================================================
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ],
    name="data_augmentation",
)

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3),
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)
outputs = tf.keras.layers.Dense(
    NUM_CLASSES, activation="softmax", name="predictions")(x)
model = tf.keras.Model(inputs, outputs, name=f"efficientnetb0_{RUN_NAME}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_fn, metrics=["accuracy"],
)

checkpoint_path = MODELS_DIR / f"{RUN_NAME}_efficientnetb0.keras"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path), monitor="val_loss",
        save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True,
        verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7,
        verbose=1),
]

# ==================================================
# 5. STAGE 1: HEAD, STAGE 2: FINE-TUNE (same as baseline)
# ==================================================
print("\nSTAGE 1: TRAINING CLASSIFICATION HEAD")
history_head = model.fit(
    train_ds, validation_data=val_ds, epochs=15,
    callbacks=callbacks, class_weight=fit_class_weight,
)
pd.DataFrame(history_head.history).to_csv(
    RESULTS_DIR / f"{RUN_NAME}_head_history.csv", index=False)

print("\nSTAGE 2: FINE-TUNING EFFICIENTNETB0")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn, metrics=["accuracy"],
)
history_finetune = model.fit(
    train_ds, validation_data=val_ds, epochs=10,
    callbacks=callbacks, class_weight=fit_class_weight,
)
pd.DataFrame(history_finetune.history).to_csv(
    RESULTS_DIR / f"{RUN_NAME}_finetune_history.csv", index=False)

# Reload best checkpoint (compile=False: custom focal loss is not needed
# for prediction).
model = tf.keras.models.load_model(checkpoint_path, compile=False)

# ==================================================
# 6. FINAL TEST EVALUATION (test set used once)
# ==================================================
print("\nFINAL TEST EVALUATION")
true_labels = np.concatenate([y.numpy() for _, y in test_ds])
probabilities = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

labels = list(range(NUM_CLASSES))
report_text = classification_report(
    true_labels, predicted_labels, labels=labels,
    target_names=class_names, zero_division=0)
report_dict = classification_report(
    true_labels, predicted_labels, labels=labels,
    target_names=class_names, output_dict=True, zero_division=0)
print(report_text)

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(RESULTS_DIR / f"{RUN_NAME}_classification_report.csv")

pd.DataFrame({
    "true_label": [class_names[i] for i in true_labels],
    "predicted_label": [class_names[i] for i in predicted_labels],
    "correct": true_labels == predicted_labels,
}).to_csv(RESULTS_DIR / f"{RUN_NAME}_test_predictions.csv", index=False)

cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
    RESULTS_DIR / f"{RUN_NAME}_confusion_matrix.csv")

plt.figure(figsize=(13, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(f"EfficientNetB0 {RUN_NAME} Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"{RUN_NAME}_confusion_matrix.png", dpi=200)
plt.close()

# ==================================================
# 7. UPDATE results/imbalance_comparison.csv
# ==================================================
def summary_row(name, rep, note):
    """One comparison row built from a classification-report table."""
    row = {
        "model": name,
        "accuracy": rep.loc["accuracy", "f1-score"],
        "macro_precision": rep.loc["macro avg", "precision"],
        "macro_recall": rep.loc["macro avg", "recall"],
        "macro_f1": rep.loc["macro avg", "f1-score"],
        "weighted_f1": rep.loc["weighted avg", "f1-score"],
        "notes": note,
    }
    for c in class_names:
        row[f"precision_{c}"] = rep.loc[c, "precision"]
    for c in class_names:
        row[f"recall_{c}"] = rep.loc[c, "recall"]
    return row


rows = []
baseline_report = RESULTS_DIR / "baseline_classification_report.csv"
if baseline_report.exists():
    base_rep = pd.read_csv(baseline_report, index_col=0)
    rows.append(summary_row("baseline", base_rep,
                            "Member 2, plain cross-entropy"))
else:
    print("\nNote: results/baseline_classification_report.csv not found.")

note = (f"class_weight={args.weight_mode}" if args.loss == "weighted"
        else f"focal gamma={args.gamma:g}"
        + (f" + class_weight={args.weight_mode}"
           if args.focal_weights else ""))
rows.append(summary_row(RUN_NAME, report_df, note))

comparison_path = RESULTS_DIR / "imbalance_comparison.csv"
new_df = pd.DataFrame(rows)
if comparison_path.exists():
    old_df = pd.read_csv(comparison_path)
    old_df = old_df[~old_df["model"].isin(new_df["model"])]
    new_df = pd.concat([old_df, new_df], ignore_index=True)
new_df.round(4).to_csv(comparison_path, index=False)

with open(RESULTS_DIR / f"{RUN_NAME}_metrics.json", "w",
          encoding="utf-8") as f:
    json.dump({
        "run": RUN_NAME, "loss": args.loss,
        "weight_mode": args.weight_mode, "gamma": args.gamma,
        "class_weights": {class_names[i]: w
                          for i, w in class_weight.items()},
        "accuracy": float(report_dict["accuracy"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
    }, f, indent=4)

print(f"\n===== {RUN_NAME} =====")
print(f"Accuracy:     {report_dict['accuracy']:.4f}")
print(f"Macro recall: {report_dict['macro avg']['recall']:.4f}")
print(f"Macro F1:     {report_dict['macro avg']['f1-score']:.4f}")
print("Saved comparison to", comparison_path)