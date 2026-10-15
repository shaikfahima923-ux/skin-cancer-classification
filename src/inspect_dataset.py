
from pathlib import Path
import csv

DATASET_ROOT = Path(
    "dataset/raw/Skin cancer ISIC The International Skin Imaging Collaboration"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder):
    return sum(
        1 for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    if not DATASET_ROOT.is_dir():
        print(f"Dataset folder not found: {DATASET_ROOT.resolve()}")
        return

    all_results = []
    total_images = 0

    for split in ["Train", "Test"]:
        split_path = DATASET_ROOT / split

        if not split_path.is_dir():
            print(f"Missing folder: {split_path}")
            continue

        print(f"\n{'=' * 45}")
        print(f"{split.upper()} DATASET")
        print("=" * 45)

        class_folders = sorted(
            folder for folder in split_path.iterdir()
            if folder.is_dir()
        )

        split_total = 0

        for folder in class_folders:
            count = count_images(folder)
            split_total += count
            all_results.append([split, folder.name, count])
            print(f"{folder.name}: {count} images")

        total_images += split_total
        print(f"\nTotal {split} images: {split_total}")
        print(f"Number of classes: {len(class_folders)}")

    print(f"\n{'=' * 45}")
    print(f"TOTAL IMAGES: {total_images}")
    print("=" * 45)

    output_path = Path("results/class_counts.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["split", "class_name", "image_count"])
        writer.writerows(all_results)

    print(f"\nSummary saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()