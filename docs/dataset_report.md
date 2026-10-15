# Dataset Inspection Report

## 1. Dataset Overview

**Dataset:** Skin cancer ISIC — The International Skin Imaging Collaboration

**Task:** Nine-class skin lesion image classification.

**Total images:** 2,357

- Training images: 2,239
- Test images: 118
- Number of classes: 9

## 2. Class-wise Image Distribution

| Class | Train | Test | Total |
|---|---:|---:|---:|
| Actinic keratosis | 114 | 16 | 130 |
| Basal cell carcinoma | 376 | 16 | 392 |
| Dermatofibroma | 95 | 16 | 111 |
| Melanoma | 438 | 16 | 454 |
| Nevus | 357 | 16 | 373 |
| Pigmented benign keratosis | 462 | 16 | 478 |
| Seborrheic keratosis | 77 | 3 | 80 |
| Squamous cell carcinoma | 181 | 16 | 197 |
| Vascular lesion | 139 | 3 | 142 |
| **Total** | **2,239** | **118** | **2,357** |

## 3. Observations

- The dataset contains nine skin lesion classes.
- The training dataset exhibits class imbalance.
- Pigmented benign keratosis has the largest number of images.
- Seborrheic keratosis has the fewest training images.
- The test set contains only three images for seborrheic keratosis and vascular lesion.

## 4. Proposed Preprocessing

- Preserve the original Train and Test splits.
- Resize images to a consistent model input size, initially 224 × 224 pixels.
- Apply model-appropriate normalization.
- Apply data augmentation only to the training data.
- Investigate class weighting to address training class imbalance.
- Evaluate using macro precision, macro recall, macro F1-score, balanced accuracy, and a confusion matrix.

## 5. Limitations

The test set is small and has very few examples for two classes. Consequently, class-specific evaluation results may be unstable. The dataset counts reflect image files detected by the inspection script; image integrity and duplicate overlap have not yet been verified.

## 6. Dataset Source

Kaggle: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

**Note:** Record the actual download date and verify image integrity before model training.
