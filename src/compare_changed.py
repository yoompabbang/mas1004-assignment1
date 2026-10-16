"""Compare a saved model's answers on original images and on changed copies.

Usage:
    uv run python src/compare_changed.py --tag clean \
        --pair data/new_images/rilakkuma data/changed/rilakkuma_gray \
        --pair data/new_images/korilakkuma data/changed/korilakkuma_gray

Each --pair is a folder of originals and a folder of changed copies with the
same file names. The true class is the name of the original folder. For every
image it prints the class the model chose with its probability, and the
probability it gave to the true class, for the original and for the copy.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from data import IMAGE_SUFFIXES, prepare_image
from train import load

ROOT = Path(__file__).resolve().parents[1]


def probabilities(model, path):
    with Image.open(path) as image:
        x = prepare_image(image)
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(x[None])), dim=1)[0].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="clean", help="which saved run to use")
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("ORIGINALS", "CHANGED"))
    args = parser.parse_args()

    settings = json.loads((ROOT / "results" / f"{args.tag}_settings.json").read_text())
    names = settings["class_names"]
    model = load(ROOT / "results" / f"{args.tag}_model.pt")

    def answer(p):
        return f"{names[p.argmax()]} ({p.max():.1%})"

    print(f"model: results/{args.tag}_model.pt")
    print("| image | true | original: said (prob) | original: true prob "
          "| changed: said (prob) | changed: true prob |")
    print("|---|---|---|---|---|---|")
    skipped = []
    for originals, changed in args.pair:
        originals, changed = Path(originals), Path(changed)
        true = names.index(originals.name)
        for path in sorted(p for p in originals.iterdir()
                           if p.suffix.lower() in IMAGE_SUFFIXES):
            if not (changed / path.name).exists():
                skipped.append(f"{originals.name}/{path.name}")
                continue
            before = probabilities(model, path)
            after = probabilities(model, changed / path.name)
            print(f"| {path.name} | {originals.name} | {answer(before)} | "
                  f"{before[true]:.1%} | {answer(after)} | {after[true]:.1%} |")
    print(f"\nchanged copies from: {', '.join(c for _, c in args.pair)}")
    if skipped:
        print(f"no changed copy, left out: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
