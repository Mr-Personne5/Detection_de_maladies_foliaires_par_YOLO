import json
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "YOLOv8n": RUNS_DIR / "train_nano" / "weights" / "best.pt",
    "YOLOv8s": RUNS_DIR / "train_small" / "weights" / "best.pt",
}


def benchmark_inference(model, n_warmup=10, n_runs=100, imgsz=640):
    dummy = torch.rand(1, 3, imgsz, imgsz).to(model.device)
    for _ in range(n_warmup):
        model.predict(dummy, verbose=False)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        model.predict(dummy, verbose=False)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) / n_runs * 1000
    return elapsed_ms


def main():
    results = {}
    for name, weights_path in MODELS.items():
        model = YOLO(str(weights_path))
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        speed_ms = benchmark_inference(model)
        n_params = sum(p.numel() for p in model.model.parameters())
        results[name] = {
            "size_mb": round(size_mb, 2),
            "inference_ms": round(speed_ms, 2),
            "params": n_params,
        }
        print(f"{name}: {size_mb:.2f} MB, {speed_ms:.2f} ms/image, {n_params:,} params")

    with open(REPORTS_DIR / "benchmark.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
