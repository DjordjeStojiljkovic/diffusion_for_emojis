"""Carve two class-balanced subsets of common emoji out of the openmoji dataset.

    python make_subsets.py

Writes datasets/openmoji_10 and datasets/openmoji_200, each laid out exactly
like the parent dataset (images/ + metadata.csv), so ConditionedEmojiDataset
reads them with no special-casing.

The picks are hand-curated rather than random: a random draw of openmoji is
mostly keycaps, clock faces and Japanese buttons, which are a poor showcase and
a strange thing to condition on. Everything below is an emoji a person would
recognise. The 10-image subset is a deliberate subset of the 200 one, so the
two runs are directly comparable.
"""

import csv
import json
import shutil
from pathlib import Path

SOURCE = Path("datasets/openmoji")
LABEL_COL = "group"

# The overfit/sanity subset: two visually unmistakable classes, five each. If
# conditioning works at all, it is obvious by eye.
SUBSET_10 = {
    "smileys-emotion": [
        "grinning face",
        "face with tears of joy",
        "smiling face with heart-eyes",
        "winking face",
        "loudly crying face",
    ],
    "food-drink": [
        "pizza",
        "hamburger",
        "birthday cake",
        "red apple",
        "hot beverage",
    ],
}

# The real run: all 8 openmoji groups, 25 common emoji each.
SUBSET_200 = {
    "smileys-emotion": [
        "grinning face", "grinning face with big eyes", "grinning face with smiling eyes",
        "face with tears of joy", "rolling on the floor laughing", "slightly smiling face",
        "upside-down face", "winking face", "smiling face with smiling eyes",
        "smiling face with halo", "smiling face with heart-eyes", "face blowing a kiss",
        "kissing face", "face with tongue", "winking face with tongue", "thinking face",
        "neutral face", "expressionless face", "smirking face", "unamused face",
        "pensive face", "sleeping face", "crying face", "loudly crying face", "angry face",
    ],
    "food-drink": [
        "red apple", "green apple", "banana", "grapes", "strawberry", "watermelon",
        "cherries", "peach", "pineapple", "carrot", "ear of corn", "broccoli", "pizza",
        "hamburger", "french fries", "hot dog", "taco", "bread", "cheese wedge", "egg",
        "popcorn", "birthday cake", "doughnut", "ice cream", "hot beverage",
    ],
    "animals-nature": [
        "dog face", "cat face", "mouse face", "rabbit face", "fox", "bear", "panda",
        "tiger face", "lion", "cow face", "pig face", "monkey face", "penguin", "bird",
        "frog", "snake", "turtle", "butterfly", "honeybee", "fish", "dolphin",
        "spouting whale", "octopus", "rose", "maple leaf",
    ],
    "travel-places": [
        "automobile", "taxi", "bus", "ambulance", "fire engine", "police car", "bicycle",
        "motorcycle", "airplane", "rocket", "helicopter", "sailboat", "ship", "train",
        "house", "office building", "school", "hospital", "castle", "mountain", "volcano",
        "sun", "full moon", "star", "rainbow",
    ],
    "objects": [
        "mobile phone", "laptop", "desktop computer", "keyboard", "camera", "television",
        "light bulb", "candle", "open book", "pencil", "paperclip", "scissors", "key",
        "locked", "hammer", "wrench", "gear", "bomb", "guitar", "microphone", "envelope",
        "package", "money bag", "credit card", "crown",
    ],
    "activities": [
        "soccer ball", "basketball", "american football", "baseball", "tennis",
        "volleyball", "bowling", "boxing glove", "trophy", "1st place medal",
        "sports medal", "video game", "joystick", "game die", "puzzle piece", "teddy bear",
        "balloon", "party popper", "confetti ball", "wrapped gift", "Christmas tree",
        "fireworks", "sparkles", "artist palette", "kite",
    ],
    "people-body": [
        "anatomical heart", "biting lip", "bone", "brain", "eye", "eyes", "lungs",
        "mechanical arm", "mechanical leg", "mouth", "tongue", "tooth", "genie", "troll",
        "zombie", "man zombie", "woman zombie", "hairy creature", "skier", "person fencing",
        "bust in silhouette", "busts in silhouette", "fingerprint", "footprints",
        "people hugging",
    ],
    "symbols": [
        "check mark button", "cross mark", "red question mark", "red exclamation mark",
        "warning", "no entry", "prohibited", "recycling symbol", "radioactive", "biohazard",
        "peace symbol", "yin yang", "latin cross", "star and crescent", "atom symbol",
        "infinity", "plus", "minus", "multiply", "divide", "red circle", "blue square",
        "up arrow", "down arrow", "play button",
    ],
}


def load_source(source=SOURCE):
    """Index the parent metadata by (group, name) — names repeat across groups."""
    with open(source / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r[LABEL_COL], r["name"]): r for r in rows}, rows[0].keys()


def build(name, picks, source=SOURCE, out_root=Path("datasets")):
    """Copy the picked emoji into datasets/<name>/ with a matching metadata.csv."""
    index, columns = load_source(source)

    missing = [(g, n) for g, names in picks.items() for n in names if (g, n) not in index]
    if missing:
        raise KeyError(f"{len(missing)} pick(s) not in {source}/metadata.csv: {missing[:5]}")

    out = out_root / name
    images = out / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)

    rows = [index[(g, n)] for g, names in sorted(picks.items()) for n in sorted(names)]
    for row in rows:
        shutil.copy2(source / "images" / row["filename"], images / row["filename"])

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)

    counts = {g: len(names) for g, names in sorted(picks.items())}
    info = {
        "name": name,
        "source": str(source),
        "num_images": len(rows),
        "label_col": LABEL_COL,
        "num_classes": len(counts),
        "class_counts": counts,
        "curated": True,
    }
    with open(out / "subset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"{name}: {len(rows)} images, {len(counts)} classes -> {out}")
    for group, count in counts.items():
        print(f"    {group:<18} {count}")
    return out


if __name__ == "__main__":
    build("openmoji_10", SUBSET_10)
    build("openmoji_200", SUBSET_200)

    # The small subset must be reproducible from the big one for the comparison
    # in the notebook to be honest.
    flat_10 = {(g, n) for g, names in SUBSET_10.items() for n in names}
    flat_200 = {(g, n) for g, names in SUBSET_200.items() for n in names}
    assert flat_10 <= flat_200, "the 10-subset should be contained in the 200-subset"
    assert len(flat_200) == 200, f"expected 200 picks, got {len(flat_200)}"
