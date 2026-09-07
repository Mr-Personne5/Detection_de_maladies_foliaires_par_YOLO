import random
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

random.seed(7)


def main():
    model = YOLO(str(RUNS_DIR / "train_small" / "weights" / "best.pt"))
    test_images = sorted((DATA_DIR / "test" / "images").glob("*"))
    sample = random.sample(test_images, 8)

    results = model.predict(source=[str(p) for p in sample], conf=0.25, verbose=False)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for ax, r in zip(axes.flatten(), results):
        ax.imshow(r.plot()[:, :, ::-1])
        ax.set_title(Path(r.path).name, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    out_path = FIG_DIR / "qualitative_yolov8s.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
