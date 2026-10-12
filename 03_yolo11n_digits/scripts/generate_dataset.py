"""Generate a reproducible multi-digit YOLO detection dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


IMAGE_SIZE = 640
CLASS_NAMES = [str(i) for i in range(10)]
SPLIT_COUNTS = {"train": 1400, "val": 400, "test": 200}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_fonts() -> list[Path]:
    fonts_dir = Path(r"C:\Windows\Fonts")
    preferred = [
        "arial.ttf", "arialbd.ttf", "ariali.ttf", "calibri.ttf", "calibrib.ttf",
        "cambria.ttc", "comic.ttf", "consola.ttf", "consolab.ttf", "cour.ttf",
        "courbd.ttf", "georgia.ttf", "impact.ttf", "segoeui.ttf", "segoeuib.ttf",
        "segoeuisl.ttf", "tahoma.ttf", "times.ttf", "timesbd.ttf", "trebuc.ttf",
        "verdanab.ttf", "verdana.ttf",
    ]
    fonts = [fonts_dir / name for name in preferred if (fonts_dir / name).is_file()]
    if len(fonts) < 8 and fonts_dir.is_dir():
        fonts.extend(path for path in sorted(fonts_dir.glob("*.ttf")) if path not in fonts)
    fonts = fonts[:24]
    if not fonts:
        raise RuntimeError("No usable Windows TrueType fonts were found.")
    return fonts


def make_background(rng: random.Random) -> Image.Image:
    mode = rng.choices(
        ["color", "gray", "white", "black", "paper"],
        weights=[35, 20, 18, 12, 15],
        k=1,
    )[0]
    if mode == "color":
        base = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        color_a = np.array([rng.randint(35, 225) for _ in range(3)], dtype=np.float32)
        color_b = np.array([rng.randint(35, 225) for _ in range(3)], dtype=np.float32)
        horizontal = np.linspace(0.0, 1.0, IMAGE_SIZE, dtype=np.float32)[None, :, None]
        gradient = color_a * (1.0 - horizontal) + color_b * horizontal
        base[:] = np.repeat(gradient, IMAGE_SIZE, axis=0).astype(np.uint8)
    elif mode == "gray":
        value = rng.randint(55, 210)
        base = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), value, dtype=np.uint8)
    elif mode == "white":
        base = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), rng.randint(220, 255), dtype=np.uint8)
    elif mode == "black":
        base = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), rng.randint(0, 35), dtype=np.uint8)
    else:
        value = rng.randint(150, 235)
        base = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), value, dtype=np.uint8)

    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(rng.randint(0, 12)):
        x0 = rng.randint(0, IMAGE_SIZE - 1)
        y0 = rng.randint(0, IMAGE_SIZE - 1)
        x1 = min(IMAGE_SIZE, x0 + rng.randint(20, 220))
        y1 = min(IMAGE_SIZE, y0 + rng.randint(20, 220))
        fill = tuple(rng.randint(0, 255) for _ in range(3)) + (rng.randint(15, 55),)
        draw.rectangle((x0, y0, x1, y1), fill=fill)
    return image


def intersection_over_area(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    return overlap / area


def yolo_line(class_id: int, box: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = box
    x_center = ((x0 + x1) / 2) / IMAGE_SIZE
    y_center = ((y0 + y1) / 2) / IMAGE_SIZE
    width = (x1 - x0) / IMAGE_SIZE
    height = (y1 - y0) / IMAGE_SIZE
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def generate_image(rng: random.Random, fonts: list[Path]) -> tuple[Image.Image, list[str]]:
    image = make_background(rng)
    labels: list[str] = []
    boxes: list[tuple[int, int, int, int]] = []
    object_count = rng.randint(1, 6)
    draw_layer = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
    background_luminance = float(np.asarray(image.convert("L"), dtype=np.float32).mean())

    for _ in range(object_count):
        digit = rng.randint(0, 9)
        font_size = rng.randint(48, 150)
        font = ImageFont.truetype(str(rng.choice(fonts)), font_size)
        glyph = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
        glyph_draw = ImageDraw.Draw(glyph)
        if background_luminance >= 155:
            palette = rng.choices(["black", "gray", "color"], weights=[55, 20, 25], k=1)[0]
        elif background_luminance <= 85:
            palette = rng.choices(["white", "gray", "color"], weights=[55, 20, 25], k=1)[0]
        else:
            palette = rng.choices(["black", "white", "gray", "color"], weights=[30, 30, 15, 25], k=1)[0]
        if palette == "black":
            foreground = (rng.randint(0, 35),) * 3 + (255,)
        elif palette == "white":
            foreground = (rng.randint(220, 255),) * 3 + (255,)
        elif palette == "gray":
            value = rng.randint(65, 190)
            foreground = (value,) * 3 + (255,)
        else:
            foreground = tuple(rng.randint(0, 255) for _ in range(3)) + (255,)
        glyph_draw.text((20, 10), str(digit), font=font, fill=foreground, stroke_width=rng.randint(0, 2))
        bbox = glyph.getbbox()
        if bbox is None:
            continue
        glyph = glyph.crop((max(0, bbox[0] - 3), max(0, bbox[1] - 3), min(glyph.width, bbox[2] + 3), min(glyph.height, bbox[3] + 3)))
        angle = rng.randint(-25, 25)
        glyph = glyph.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        width, height = glyph.size

        placed = False
        for _attempt in range(50):
            x0 = rng.randint(8, max(8, IMAGE_SIZE - width - 8))
            y0 = rng.randint(8, max(8, IMAGE_SIZE - height - 8))
            candidate = (x0, y0, min(IMAGE_SIZE, x0 + width), min(IMAGE_SIZE, y0 + height))
            if all(intersection_over_area(candidate, old) < 0.20 for old in boxes):
                placed = True
                break
        if not placed:
            continue

        draw_layer.alpha_composite(glyph, (x0, y0))
        boxes.append(candidate)
        labels.append(yolo_line(digit, candidate))

    image = Image.alpha_composite(image.convert("RGBA"), draw_layer).convert("RGB")
    if rng.random() < 0.45:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.35))
    if rng.random() < 0.45:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.30:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.1)))
    if rng.random() < 0.35:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(2, 9), (IMAGE_SIZE, IMAGE_SIZE, 3))
        pixels = np.clip(np.asarray(image, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
    return image, labels


def main() -> int:
    args = parse_args()
    if args.count != sum(SPLIT_COUNTS.values()):
        raise SystemExit(f"This first reproducible run expects count={sum(SPLIT_COUNTS.values())}.")
    rng = random.Random(args.seed)
    fonts = find_fonts()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_COUNTS:
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    index = 0
    class_counts = {name: 0 for name in CLASS_NAMES}
    for split, split_count in SPLIT_COUNTS.items():
        for _ in range(split_count):
            image, labels = generate_image(rng, fonts)
            if not labels:
                raise RuntimeError("Generated an image without any labeled digit.")
            stem = f"digit_{index:05d}"
            image.save(args.output / "images" / split / f"{stem}.png", optimize=True)
            (args.output / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            for line in labels:
                class_counts[line.split()[0]] += 1
            index += 1

    summary = {
        "seed": args.seed,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "splits": SPLIT_COUNTS,
        "classes": CLASS_NAMES,
        "class_instance_counts": class_counts,
        "fonts": [str(font) for font in fonts],
    }
    (args.output / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
