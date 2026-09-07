import argparse
import time
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", default="ram", choices=["ram", "disk", "none"])
    args = parser.parse_args()

    model = YOLO(args.model)
    start = time.time()
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(RUNS_DIR),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
    )
    elapsed = time.time() - start
    print(f"TOTAL TRAIN TIME: {elapsed:.1f}s ({elapsed/60:.1f} min) for {args.epochs} epochs")


if __name__ == "__main__":
    main()
