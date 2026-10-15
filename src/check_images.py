
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path(
    "dataset/raw/Skin cancer ISIC The International Skin Imaging Collaboration"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not DATASET_ROOT.is_dir():
        print(f"Dataset folder not found: {DATASET_ROOT.resolve()}")
        return

    total_checked = 0
    unreadable = []

    for split in ["Train", "Test"]:
        split_path = DATASET_ROOT / split

        if not split_path.is_dir():
            print(f"Missing split: {split}")
            continue

        print(f"\nChecking {split} images...")

        for file_path in split_path.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            total_checked += 1

            try:
                with Image.open(file_path) as image:
                    image.verify()
            except Exception as error:
                unreadable.append((str(file_path), str(error)))
                print(f"Unreadable: {file_path}")

    print("\n========== RESULTS ==========")
    print(f"Images checked: {total_checked}")
    print(f"Unreadable images: {len(unreadable)}")

    report_path = Path("results/image_integrity_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as report:
        report.write(f"Images checked: {total_checked}\n")
        report.write(f"Unreadable images: {len(unreadable)}\n\n")

        for file_path, error in unreadable:
            report.write(f"{file_path}\nError: {error}\n\n")

    print(f"Report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
