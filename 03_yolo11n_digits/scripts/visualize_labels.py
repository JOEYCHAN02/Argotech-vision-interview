"""Draw sampled YOLO labels back onto images for a visual dataset audit."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


CLASS_NAMES = [str(index) for index in range(10)]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = [
    (255, 99, 71),
    (50, 205, 50),
    (30, 144, 255),
    (255, 165, 0),
    (186, 85, 211),
    (0, 206, 209),
    (255, 215, 0),
    (147, 112, 219),
    (0, 191, 255),
    (255, 105, 180),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("results/label_previews"))
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--count", type=int, default=50, help="Number of images to sample.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_images(dataset: Path, split: str) -> list[tuple[str, Path, Path]]:
    splits = ("train", "val", "test") if split == "all" else (split,)
    items: list[tuple[str, Path, Path]] = []
    for current_split in splits:
        image_dir = dataset / "images" / current_split
        label_dir = dataset / "labels" / current_split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            items.append((current_split, image_path, label_path))
    return items


def read_labels(label_path: Path, width: int, height: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    labels: list[tuple[int, tuple[int, int, int, int]]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = raw_line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain 5 values")
        class_id, x_center, y_center, box_width, box_height = map(float, values)
        if not class_id.is_integer() or not 0 <= int(class_id) < len(CLASS_NAMES):
            raise ValueError(f"{label_path}:{line_number} has invalid class id")
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, box_width, box_height)):
            raise ValueError(f"{label_path}:{line_number} has a coordinate outside [0, 1]")

        x0 = max(0, round((x_center - box_width / 2) * width))
        y0 = max(0, round((y_center - box_height / 2) * height))
        x1 = min(width - 1, round((x_center + box_width / 2) * width))
        y1 = min(height - 1, round((y_center + box_height / 2) * height))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"{label_path}:{line_number} has an empty box")
        labels.append((int(class_id), (x0, y0, x1, y1)))
    return labels


def draw_labels(image, labels: list[tuple[int, tuple[int, int, int, int]]]):
    for class_id, (x0, y0, x1, y1) in labels:
        color = COLORS[class_id]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        text = CLASS_NAMES[class_id]
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        label_top = max(0, y0 - text_height - baseline - 4)
        cv2.rectangle(image, (x0, label_top), (x0 + text_width + 6, y0), color, -1)
        cv2.putText(
            image,
            text,
            (x0 + 3, y0 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")

    items = collect_images(args.dataset, args.split)
    if len(items) < args.count:
        raise SystemExit(f"Requested {args.count} images, but only {len(items)} are available.")

    rng = random.Random(args.seed)
    selected = rng.sample(items, args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# Label preview report",
        "",
        f"- Split selection: `{args.split}`",
        f"- Sample count: `{args.count}`",
        f"- Random seed: `{args.seed}`",
        "- Output: images with ground-truth YOLO boxes drawn in color.",
        "",
        "| # | Split | Source image | Label file | Objects |",
        "|---:|---|---|---|---:|",
    ]

    for index, (split, image_path, label_path) in enumerate(selected, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unable to read image: {image_path}")
        height, width = image.shape[:2]
        labels = read_labels(label_path, width, height)
        preview = draw_labels(image, labels)
        output_path = args.output / f"{index:03d}_{split}_{image_path.name}"
        if not cv2.imwrite(str(output_path), preview):
            raise RuntimeError(f"Unable to write preview: {output_path}")
        report_lines.append(
            f"| {index} | `{split}` | `{image_path.name}` | `{label_path.name}` | {len(labels)} |"
        )

    report_path = args.output / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Created {len(selected)} previews in {args.output.resolve()}")
    print(f"Wrote report to {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
