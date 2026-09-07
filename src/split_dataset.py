"""
Split le dossier data/train (fourni tel quel par l'export Roboflow, sans
valid/test) en train/valid/test (80/10/10), par déplacement de fichiers.
"""
import random
import shutil
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SRC_IMAGES = DATA_DIR / "train" / "images"
SRC_LABELS = DATA_DIR / "train" / "labels"

TRAIN_RATIO, VALID_RATIO = 0.8, 0.1  # le reste (0.1) va en test

def main():
    image_files = sorted(SRC_IMAGES.glob("*"))
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    splits = {
        "valid": image_files[n_train:n_train + n_valid],
        "test": image_files[n_train + n_valid:],
    }

    for split_name, files in splits.items():
        img_dst = DATA_DIR / split_name / "images"
        lbl_dst = DATA_DIR / split_name / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            label_path = SRC_LABELS / (img_path.stem + ".txt")
            shutil.move(str(img_path), str(img_dst / img_path.name))
            if label_path.exists():
                shutil.move(str(label_path), str(lbl_dst / label_path.name))

    n_train_final = len(list(SRC_IMAGES.glob("*")))
    print(f"train: {n_train_final} images")
    for split_name in splits:
        n_split = len(list((DATA_DIR / split_name / "images").glob("*")))
        print(f"{split_name}: {n_split} images")

if __name__ == "__main__":
    main()
