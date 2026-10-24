# Class Imbalance Handling: Report (Member 3A)

## 1. Goal
Improve EfficientNetB0 performance on underrepresented skin lesion classes and compare against Member 2's baseline.

## 2. Class distribution (training subset, 80% of Train)
Counts and weights come from training labels only. The same split as the baseline is used (validation_split=0.20, seed=42). The full table is in `results/imbalance_class_counts.csv`.

| Class | Train images | Share | Balanced weight |
|---|---|---|---|
| pigmented benign keratosis | 369 | 20.6% | 0.54 |
| nevus | 286 | 16.0% | 0.70 |
| squamous cell carcinoma | 142 | 7.9% | 1.40 |
| vascular lesion | 107 | 6.0% | 1.86 |
| seborrheic keratosis | 61 | 3.4% | 3.26 |
| actinic keratosis | 96 | 5.4% | 2.07 |
| basal cell carcinoma | 310 | 17.3% | 0.64 |
| dermatofibroma | 78 | 4.4% | 2.55 |
| melanoma | 343 | 19.1% | 0.58 |

Weight = n_samples / (n_classes x class_count). Rare classes get larger weights.

## 3. Method
- Same pipeline as the baseline: EfficientNetB0, 224x224 input, batch size 16, same augmentation, two-stage training (head, then last 30 layers), same callbacks.
- Only change: Keras `class_weight` with balanced weights during `model.fit`.
- Validation split used for checkpointing; the official test set was used once for final evaluation.
- Code: `src/class_weights.py`, `src/train_weighted.py`.

## 4. Overall results (official test set, 118 images)

| Metric | Baseline | Weighted | Change |
|---|---|---|---|
| Accuracy | 0.398 | 0.449 | +0.051 |
| Macro precision | 0.447 | 0.483 | +0.036 |
| Macro recall | 0.417 | 0.458 | +0.042 |
| Macro F1 | 0.365 | 0.432 | +0.067 |
| Weighted F1 | 0.363 | 0.432 | +0.069 |

## 5. Per-class results

| Class | Test n | Recall base | Recall weighted | Precision base | Precision weighted |
|---|---|---|---|---|---|
| actinic keratosis | 16 | 0.12 | 0.25 | 0.67 | 0.67 |
| basal cell carcinoma | 16 | 0.31 | 0.31 | 0.42 | 0.56 |
| dermatofibroma | 16 | 0.25 | 0.38 | 1.00 | 0.86 |
| melanoma | 16 | 0.19 | 0.19 | 0.23 | 0.27 |
| nevus | 16 | 0.88 | 0.88 | 0.41 | 0.44 |
| pigmented benign keratosis | 16 | 0.62 | 0.56 | 0.32 | 0.38 |
| seborrheic keratosis | 3 | 0.00 | 0.00 | 0.00 | 0.00 |
| squamous cell carcinoma | 16 | 0.38 | 0.56 | 0.38 | 0.43 |
| vascular lesion | 3 | 1.00 | 1.00 | 0.60 | 0.75 |

The confusion matrix is in `results/weighted_balanced_confusion_matrix.png`.

## 6. Findings
- Class weighting improved every overall metric. Macro F1 rose from 0.365 to 0.432.
- Recall improved for actinic keratosis (+0.13), dermatofibroma (+0.13) and squamous cell carcinoma (+0.18).
- Precision improved for basal cell carcinoma, squamous cell carcinoma and vascular lesion. It fell for dermatofibroma (1.00 to 0.86).
-- Melanoma recall stayed low at 0.19 even though it has 343 training images (19%). The problem is not scarcity, so weighting could not fix it. It is probably visual similarity to nevus and other benign lesions. Seborrheic keratosis (61 images, only 3 test images) stayed at 0.00.
- Pigmented benign keratosis recall dropped slightly (0.62 to 0.56), a small trade-off from shifting predictions toward rare classes.

## 7. Limitations
- The test set is small (118 images; 16 per class, only 3 for seborrheic keratosis and vascular lesion). One image changes recall by about 0.06 (0.33 for the 3-image classes), so per-class differences are noisy.
- Results come from a single run with one seed.
- Checkpoints are selected by validation loss, which can favour large classes.

## 8. Next steps
- Try square-root weights: `python src/train_weighted.py --weight-mode sqrt`.
- Try focal loss: `python src/train_weighted.py --loss focal --gamma 2.0`.
- Repeat with several seeds to check the gain is stable.
-Investigate melanoma confusions using the confusion matrix, and try targeted augmentation or a higher-resolution input