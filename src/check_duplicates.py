
from pathlib import Path
import hashlib
from collections import defaultdict

DATASET_ROOT = Path(
    "dataset/raw/Skin cancer ISIC The International Skin Imaging Collaboration"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_hash(file_path):
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    split_hashes = {"Train": defaultdict(list), "Test": defaultdict(list)}

    for split in ["Train", "Test"]:
        split_path = DATASET_ROOT / split

        if not split_path.is_dir():
            print(f"Missing folder: {split_path}")
            return

        print(f"Scanning {split}...")

        for file_path in split_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    file_hash = get_hash(file_path)
                    split_hashes[split][file_hash].append(str(file_path))
                except OSError as error:
                    print(f"Could not read {file_path}: {error}")

    train_hashes = set(split_hashes["Train"])
    test_hashes = set(split_hashes["Test"])
    shared_hashes = train_hashes & test_hashes

    print("\n========== DUPLICATE CHECK ==========")
    print(f"Unique Train image contents: {len(train_hashes)}")
    print(f"Unique Test image contents: {len(test_hashes)}")
    print(f"Exact duplicate contents across splits: {len(shared_hashes)}")

    report_path = Path("results/duplicate_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as report:
        report.write(f"Unique Train image contents: {len(train_hashes)}\n")
        report.write(f"Unique Test image contents: {len(test_hashes)}\n")
        report.write(f"Exact duplicate contents across splits: {len(shared_hashes)}\n\n")

        for file_hash in sorted(shared_hashes):
            report.write("Matching hash: " + file_hash + "\n")
            report.write("Train files:\n")
            for path in split_hashes["Train"][file_hash]:
                report.write(f"  {path}\n")
            report.write("Test files:\n")
            for path in split_hashes["Test"][file_hash]:
                report.write(f"  {path}\n")
            report.write("\n")

    print(f"Report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
