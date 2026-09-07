from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "data.yaml") as f:
    CLASS_NAMES = yaml.safe_load(f)["names"]

SELECTED = [
    "IMG_20221118_102314_482_jpg.rf.565535616f58fd0ea5c4cef62bbd360b.jpg",
    "Manioc_Mosaique_-150-_jpg.rf.89c1825c1bfb01db34d348feb0256776.jpg",
    "Mais_Brulure_-255-_jpg.rf.fc0744a0143c9dee821ba9300b5011c5.jpg",
]


def draw_gt(ax, img_path):
    img = Image.open(img_path)
    w, h = img.size
    ax.imshow(img)
    label_path = DATA_DIR / "test" / "labels" / (img_path.stem + ".txt")
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                if not line.strip():
                    continue
                cls_id, cx, cy, bw, bh = map(float, line.split())
                cls_id = int(cls_id)
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                rect = patches.Rectangle((x, y), bw * w, bh * h, linewidth=1.5, edgecolor="red", facecolor="none")
                ax.add_patch(rect)
                ax.text(x, y - 5, CLASS_NAMES[cls_id], color="red", fontsize=7, weight="bold")
    ax.axis("off")


def main():
    model = YOLO(str(RUNS_DIR / "train_small" / "weights" / "best.pt"))

    fig, axes = plt.subplots(len(SELECTED), 2, figsize=(14, 6 * len(SELECTED)))
    for i, name in enumerate(SELECTED):
        img_path = DATA_DIR / "test" / "images" / name
        draw_gt(axes[i, 0], img_path)
        axes[i, 0].set_title(f"Vérité terrain: {name}", fontsize=8)

        result = model.predict(source=str(img_path), conf=0.25, verbose=False)[0]
        axes[i, 1].imshow(result.plot()[:, :, ::-1])
        axes[i, 1].set_title("Prédiction YOLOv8s", fontsize=8)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = FIG_DIR / "gt_vs_pred.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
