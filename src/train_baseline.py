
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

# ==================================================
# 1. PROJECT PATHS AND SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    ROOT / "dataset" / "raw"
    / "Skin cancer ISIC The International Skin Imaging Collaboration"
)

TRAIN_DIR = DATA_ROOT / "Train"
TEST_DIR = DATA_ROOT / "Test"

RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
NUM_CLASSES = 9

tf.keras.utils.set_random_seed(SEED)

# Check that dataset folders exist.
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")


# ==================================================
# 2. LOAD TRAINING AND VALIDATION DATA
# ==================================================

# Use 80% of Train for training.
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
)

# Use the remaining 20% of Train for validation.
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

# Official test set is reserved for final evaluation.
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_ds.class_names

print("\nCLASS ORDER:")
for index, name in enumerate(class_names):
    print(index, ":", name)

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected 9 classes, but found {len(class_names)}"
    )

if class_names != val_ds.class_names:
    raise ValueError("Training and validation class orders differ.")

if class_names != test_ds.class_names:
    raise ValueError(
        "Training and test class orders differ. Check class folders."
    )

# Improve input-pipeline performance.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ==================================================
# 3. DATA AUGMENTATION
# ==================================================

# Augmentation runs during training, not evaluation.
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ],
    name="data_augmentation",
)


# ==================================================
# 4. LOAD PRETRAINED EFFICIENTNETB0
# ==================================================

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

# Stage 1: freeze pretrained feature-extraction layers.
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)

# Keras EfficientNetB0 includes its own input rescaling.
# Do not divide images by 255 here.
x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="predictions",
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="efficientnetb0_baseline",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nMODEL SUMMARY:")
model.summary()


# ==================================================
# 5. TRAINING CALLBACKS
# ==================================================

checkpoint_path = MODELS_DIR / "baseline_efficientnetb0.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]


# ==================================================
# 6. STAGE 1: TRAIN CLASSIFICATION HEAD
# ==================================================

print("\nSTAGE 1: TRAINING CLASSIFICATION HEAD")

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
)

# Save Stage 1 history.
pd.DataFrame(history_head.history).to_csv(
    RESULTS_DIR / "baseline_head_history.csv",
    index=False,
)


# ==================================================
# 7. STAGE 2: FINE-TUNE UPPER LAYERS
# ==================================================

print("\nSTAGE 2: FINE-TUNING EFFICIENTNETB0")

base_model.trainable = True

# Train only the last 30 backbone layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen for stability.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Recompile after changing trainable layers.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
)

pd.DataFrame(history_finetune.history).to_csv(
    RESULTS_DIR / "baseline_finetune_history.csv",
    index=False,
)


# ==================================================
# 8. LOAD BEST MODEL AND SAVE
# ==================================================

# The checkpoint is selected using validation loss.
model = tf.keras.models.load_model(checkpoint_path)

# Save the best model at the expected path.
model.save(checkpoint_path)

print("\nBest model saved to:", checkpoint_path)


# ==================================================
# 9. FINAL EVALUATION ON OFFICIAL TEST SET
# ==================================================

print("\nFINAL TEST EVALUATION")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1,
)

true_labels = np.concatenate(
    [labels.numpy() for _, labels in test_ds]
)

probabilities = model.predict(test_ds, verbose=1)

predicted_labels = np.argmax(probabilities, axis=1)

macro_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)

report_text = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    zero_division=0,
)

report_dict = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

print("\nCLASSIFICATION REPORT:")
print(report_text)

metrics = {
    "model": "EfficientNetB0",
    "method": "Transfer learning and fine-tuning",
    "image_size": "224x224",
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "class_names": class_names,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "test_macro_f1": float(macro_f1),
    "test_weighted_f1": float(weighted_f1),
}

with open(
    RESULTS_DIR / "baseline_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

pd.DataFrame(report_dict).transpose().to_csv(
    RESULTS_DIR / "baseline_classification_report.csv"
)

predictions_df = pd.DataFrame(
    {
        "true_label": [class_names[i] for i in true_labels],
        "predicted_label": [
            class_names[i] for i in predicted_labels
        ],
        "correct": true_labels == predicted_labels,
    }
)

predictions_df.to_csv(
    RESULTS_DIR / "baseline_test_predictions.csv",
    index=False,
)


# ==================================================
# 10. CREATE CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(NUM_CLASSES)),
)

plt.figure(figsize=(13, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("EfficientNetB0 Baseline Confusion Matrix")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "baseline_confusion_matrix.png",
    dpi=200,
)

plt.close()


# ==================================================
# 11. PRINT FINAL RESULTS
# ==================================================

print("\n========== BASELINE RESULTS ==========")
print(f"Test loss:       {test_loss:.4f}")
print(f"Test accuracy:   {test_accuracy:.4f}")
print(f"Macro F1-score:  {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")

print("\nSaved output files:")
print("Model:             ", checkpoint_path)
print("Metrics:           ", RESULTS_DIR / "baseline_metrics.json")
print("Classification:    ", RESULTS_DIR / "baseline_classification_report.csv")
print("Predictions:       ", RESULTS_DIR / "baseline_test_predictions.csv")
print("Confusion matrix:  ", RESULTS_DIR / "baseline_confusion_matrix.png")
print("Training history:  ", RESULTS_DIR / "baseline_head_history.csv")
print("Fine-tune history: ", RESULTS_DIR / "baseline_finetune_history.csv")

print("\nBaseline training and evaluation completed.")
