
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

DATASET_ROOT = Path(
    "dataset/raw/Skin cancer ISIC The International Skin Imaging Collaboration"
)

CLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
]

IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_split(split_name):
    """
    Load one split and return:
      images: NumPy array shaped (N, 224, 224, 3), values in [0, 1]
      labels: NumPy array of integer class IDs
    """
    split_path = DATASET_ROOT / split_name

    if not split_path.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    images = []
    labels = []
    skipped = []
    counts = Counter()

    for label, class_name in enumerate(CLASS_NAMES):
        class_path = split_path / class_name

        if not class_path.is_dir():
            raise FileNotFoundError(
                f"Class folder not found: {class_path}"
            )

        for image_path in sorted(class_path.rglob("*")):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image = image.resize(IMAGE_SIZE)
                    array = np.asarray(image, dtype=np.float32) / 255.0

                images.append(array)
                labels.append(label)
                counts[class_name] += 1

            except (OSError, ValueError) as error:
                skipped.append((str(image_path), str(error)))

    if not images:
        raise ValueError(f"No readable images found in {split_name}")

    x = np.stack(images)
    y = np.asarray(labels, dtype=np.int64)

    print(f"\n{split_name}: {len(x)} images loaded")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {counts[class_name]}")
    print(f"Skipped unreadable images: {len(skipped)}")

    return x, y


if __name__ == "__main__":
    X_train, y_train = load_split("Train")
    X_test, y_test = load_split("Test")

    print("\n========== LOADER SUMMARY ==========")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Class names:", CLASS_NAMES)
