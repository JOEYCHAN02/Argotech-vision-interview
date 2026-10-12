"""Train the YOLO11n digit detector with reproducible project settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data.yaml")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "yolo11n.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="formal_yolo11n_100e")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration was not found: {data_path}")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        raise ValueError("epochs and batch must be positive; imgsz must be at least 32")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested, but PyTorch cannot access a GPU")

    output_root = (PROJECT_ROOT / "runs").resolve()
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(output_root),
        name=args.name,
        exist_ok=False,
        plots=True,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
