"""Compare saved runs on the images from a new source, class by class.

Usage:
    uv run python src/compare_runs.py --tags run clean --show rilakkuma

For every class in data/new_images it prints how many images each run got
right. For the classes named with --show it also prints, image by image, the
class each run chose and the probability it gave to that class.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from data import IMAGE_SUFFIXES, prepare_image
from train import load

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data" / "new_images"


def probabilities(model, path):
    with Image.open(path) as image:
        x = prepare_image(image)
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(x[None])), dim=1)[0].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=["run", "clean"])
    parser.add_argument("--show", nargs="*", default=[],
                        help="classes to list image by image")
    args = parser.parse_args()

    images = {d.name: sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
              for d in sorted(NEW.iterdir()) if d.is_dir()}

    answers = {}  # tag -> path -> (name it chose, probability)
    right = {}    # tag -> class -> count
    for tag in args.tags:
        names = json.loads((ROOT / "results" / f"{tag}_settings.json").read_text())["class_names"]
        model = load(ROOT / "results" / f"{tag}_model.pt")
        answers[tag], right[tag] = {}, {}
        for name, paths in images.items():
            right[tag][name] = 0
            for path in paths:
                p = probabilities(model, path)
                said = names[p.argmax()]
                answers[tag][path] = (said, float(p.max()))
                right[tag][name] += said == name

    print("right answers on data/new_images, per class")
    print(f"  {'class':16s} {'images':>6s}" + "".join(f" {t:>10s}" for t in args.tags))
    for name, paths in images.items():
        print(f"  {name:16s} {len(paths):6d}"
              + "".join(f" {right[t][name]:>6d}/{len(paths):<3d}" for t in args.tags))
    total = sum(len(p) for p in images.values())
    print(f"  {'total':16s} {total:6d}"
          + "".join(f" {sum(right[t].values()):>6d}/{total:<3d}" for t in args.tags))

    for name in args.show:
        print(f"\n{name}, image by image (the class each run chose, and its probability)")
        print(f"  {'image':12s}" + "".join(f" {t:>26s}" for t in args.tags))
        for path in images[name]:
            print(f"  {path.name:12s}" + "".join(
                f" {answers[t][path][0]:>18s} ({answers[t][path][1]:5.1%})" for t in args.tags))


if __name__ == "__main__":
    main()
