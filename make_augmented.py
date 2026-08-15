"""Build augmented OpenMoji datasets from all 1321 emoji.

Two datasets come out of one pass, because the augmentations are not
interchangeable between the two conditioning schemes:

    datasets/openmoji_aug_geom/    geometry only    -> text-conditional
    datasets/openmoji_aug_color/   geometry+colour  -> class-conditional

The text path embeds the emoji's own `name` + `description` with CLIP, and
plenty of those names carry a colour ("red heart", "green apple").  Hue-shifting
the pixels while keeping the caption would make the training pair a lie, so the
text dataset gets geometry only.  The class path labels by broad `group`, which
has nothing to do with hue, so colour jitter there is free regularisation.

Augmentations are rendered from OpenMoji's 618x618 originals rather than from
the 64x64 exports, so a rotation is a genuine re-render instead of a resample of
an already-tiny image.  The originals are cached under datasets/.cache_618/.

Run from the repo root::

    python make_augmented.py --limit 20 --variants 4 --out-prefix _smoketest
    python make_augmented.py
    python make_augmented.py --verify

Both datasets share one RNG stream per (emoji, variant), so `_geom/X_a03.png`
and `_color/X_a03.png` have identical geometry and differ only in colour.
"""

import argparse
import csv
import json
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
DATASETS = PROJECT_DIR / "datasets"
SOURCE = DATASETS / "openmoji"
CACHE_618 = DATASETS / ".cache_618"

IMAGE_URL = (
    "https://raw.githubusercontent.com/"
    "hfg-gmuend/openmoji/master/color/618x618/{code}.png"
)

# Augmentation ranges.  Kept deliberately mild: at 64x64 a pictogram has very
# few pixels to spare, and an aggressive crop or rotation destroys the glyph
# rather than presenting a new view of it.
ROTATION_DEG = 12.0            # +/- degrees
SCALE_RANGE = (0.85, 1.10)     # fraction of the side kept; >1 zooms out
HUE_SHIFT = 40                 # +/- units on PIL's 0-255 hue scale (~56 deg)
SATURATION_RANGE = (0.6, 1.4)
BRIGHTNESS_RANGE = (0.85, 1.15)
CONTRAST_RANGE = (0.9, 1.15)

# The eight source columns are kept verbatim so nothing downstream has to change;
# the last three are new and exist purely for auditing what was applied.
CSV_COLUMNS = [
    "filename", "emoji", "code", "name", "description", "group", "subgroup",
    "split", "aug_index", "source_filename", "aug_ops",
]

WHITE = (255, 255, 255)


# --------------------------------------------------------------------------
# source metadata + full-resolution cache
# --------------------------------------------------------------------------

def read_source_rows(limit=None):
    """The curated 1321-emoji table: already free of skin tones and flags."""
    csv_path = SOURCE / "metadata.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    # Reading the codes off this CSV rather than re-filtering openmoji.json
    # guarantees the augmented sets cover exactly the same emoji as the
    # existing 64x64 export.
    rows.sort(key=lambda row: row["code"])
    return rows[:limit] if limit else rows


def download_cache(rows, delay=0.02):
    """Fetch the 618x618 originals, skipping any already on disk."""
    CACHE_618.mkdir(parents=True, exist_ok=True)
    missing = [r for r in rows if not (CACHE_618 / f"{r['code']}.png").exists()]
    if not missing:
        return

    session = requests.Session()
    for row in tqdm(missing, desc="618x618 originals"):
        code = row["code"]
        try:
            response = session.get(IMAGE_URL.format(code=code), timeout=60)
            response.raise_for_status()
            (CACHE_618 / f"{code}.png").write_bytes(response.content)
            time.sleep(delay)
        except Exception as error:
            print(f"\nWARNING: could not fetch {code}: {error}")


def load_base(code):
    """Return (rgb, alpha, fell_back) at full source resolution.

    The RGB is already composited over white, so every later resample blends
    towards white instead of towards the black that sits in OpenMoji's fully
    transparent pixels.  The alpha is kept separately as a *mask*, not for
    compositing: it is what protects the background from the brightness and
    contrast enhancers, which would otherwise turn white into grey.
    """
    cached = CACHE_618 / f"{code}.png"
    if cached.exists():
        try:
            source = Image.open(cached).convert("RGBA")
            background = Image.new("RGBA", source.size, (255, 255, 255, 255))
            background.alpha_composite(source)
            return background.convert("RGB"), source.getchannel("A"), False
        except Exception as error:
            print(f"\nWARNING: unreadable cache for {code}: {error}")

    # Fall back to upsampling the existing 64x64 export.  Lossy, but better
    # than dropping the emoji, and recorded in dataset_info.json.
    fallback = SOURCE / "images" / f"{code}.png"
    rgb = Image.open(fallback).convert("RGB")
    alpha = Image.new("L", rgb.size, 255)
    return rgb, alpha, True


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------

def sample_params(code, variant, seed, hflip):
    """Draw one variant's parameters.

    Geometry and colour use separate RNG streams so that building the colour
    dataset does not shift the geometry draws: variant N is the same shape in
    both datasets, which makes the colour ablation a clean comparison.
    """
    if variant == 0:
        return {"angle": 0.0, "scale": 1.0, "dx": 0.0, "dy": 0.0, "flip": False,
                "hue": 0, "sat": 1.0, "bright": 1.0, "contrast": 1.0,
                "identity": True}

    geom = random.Random(f"{seed}:{code}:{variant}:geom")
    colour = random.Random(f"{seed}:{code}:{variant}:colour")

    scale = geom.uniform(*SCALE_RANGE)
    return {
        "angle": geom.uniform(-ROTATION_DEG, ROTATION_DEG),
        "scale": scale,
        # Offsets are expressed as a fraction of the slack left by the crop, so
        # they can never push the window past what `scale` already allowed.
        "dx": geom.uniform(0.0, 1.0),
        "dy": geom.uniform(0.0, 1.0),
        "flip": hflip and geom.random() < 0.5,
        "hue": colour.randint(-HUE_SHIFT, HUE_SHIFT),
        "sat": colour.uniform(*SATURATION_RANGE),
        "bright": colour.uniform(*BRIGHTNESS_RANGE),
        "contrast": colour.uniform(*CONTRAST_RANGE),
        "identity": False,
    }


def apply_colour(rgb, alpha, params):
    """Recolour the glyph, then force the background back to pure white.

    Order matters.  `ImageEnhance.Brightness(0.9)` applied to a flattened image
    turns the white background into 229-grey; the diffusion model then spends
    capacity modelling background variation and samples come out muddy.
    Compositing against white through the alpha mask afterwards keeps every
    output image's background exactly (255, 255, 255) while still blending
    antialiased edges correctly.
    """
    if params["hue"]:
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.uint8).copy()
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + params["hue"]) % 256
        rgb = Image.fromarray(hsv, "HSV").convert("RGB")

    rgb = ImageEnhance.Color(rgb).enhance(params["sat"])
    rgb = ImageEnhance.Brightness(rgb).enhance(params["bright"])
    rgb = ImageEnhance.Contrast(rgb).enhance(params["contrast"])

    white = Image.new("RGB", rgb.size, WHITE)
    return Image.composite(rgb, white, alpha)


def apply_geometry(rgb, params, out_size):
    """Rotate, then crop-or-pad, resampling straight down to `out_size`.

    Fills are white rather than transparent, which is safe because the image is
    already composited over white -- so a rotated corner and a padded margin are
    indistinguishable from the real background.
    """
    if params["flip"]:
        rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if params["angle"]:
        rgb = rgb.rotate(
            params["angle"], resample=Image.Resampling.BICUBIC, fillcolor=WHITE
        )

    width, height = rgb.size
    box_w = round(width * params["scale"])
    box_h = round(height * params["scale"])
    x0 = round((width - box_w) * params["dx"])
    y0 = round((height - box_h) * params["dy"])

    if params["scale"] <= 1.0:
        # Crop and downsample in a single resample.
        return rgb.resize(
            (out_size, out_size),
            Image.Resampling.LANCZOS,
            box=(x0, y0, x0 + box_w, y0 + box_h),
        )

    # Zoomed out: the window is larger than the image, so paste onto white.
    canvas = Image.new("RGB", (box_w, box_h), WHITE)
    canvas.paste(rgb, (-x0, -y0))
    return canvas.resize((out_size, out_size), Image.Resampling.LANCZOS)


def describe(params, with_colour):
    if params["identity"]:
        return "original"
    parts = [f"rot={params['angle']:+.1f}", f"scale={params['scale']:.3f}"]
    if params["flip"]:
        parts.append("hflip")
    if with_colour:
        parts += [
            f"hue={params['hue']:+d}",
            f"sat={params['sat']:.2f}",
            f"bri={params['bright']:.2f}",
            f"con={params['contrast']:.2f}",
        ]
    return ";".join(parts)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def prepare_output(root):
    images = root / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True, exist_ok=True)
    return images


def build(args):
    rows = read_source_rows(args.limit)
    print(f"{len(rows)} source emoji")

    if not args.no_download:
        download_cache(rows)

    targets = {
        "geom": DATASETS / f"{args.out_prefix}_geom",
        "color": DATASETS / f"{args.out_prefix}_color",
    }
    image_dirs = {key: prepare_output(root) for key, root in targets.items()}
    records = {key: [] for key in targets}
    fallbacks = []

    for row in tqdm(rows, desc="augmenting"):
        code = row["code"]
        base_rgb, base_alpha, fell_back = load_base(code)
        if fell_back:
            fallbacks.append(code)

        # Everything downstream of the first resample happens at --working-size;
        # 4x linear oversample over the 64px target is ample and much cheaper
        # than rotating at 618.
        work = (args.working_size, args.working_size)
        work_rgb = base_rgb.resize(work, Image.Resampling.LANCZOS)
        work_alpha = base_alpha.resize(work, Image.Resampling.LANCZOS)

        for variant in range(args.variants):
            params = sample_params(code, variant, args.seed, args.hflip)
            filename = f"{code}_a{variant:02d}.png"

            for key, image_dir in image_dirs.items():
                if params["identity"]:
                    # Straight from full resolution, single resample -- so a00
                    # reproduces datasets/openmoji/images/ exactly.
                    out = base_rgb.resize(
                        (args.image_size, args.image_size),
                        Image.Resampling.LANCZOS,
                    )
                else:
                    rgb = work_rgb
                    if key == "color":
                        rgb = apply_colour(rgb, work_alpha, params)
                    out = apply_geometry(rgb, params, args.image_size)

                out.save(image_dir / filename)

                record = {column: row.get(column, "") for column in CSV_COLUMNS}
                record.update({
                    "filename": filename,
                    "aug_index": variant,
                    "source_filename": row["filename"],
                    # `split` is inherited, so all variants of one emoji land on
                    # the same side of the split and no augmented copy of a
                    # train emoji can leak into val.
                    "aug_ops": describe(params, with_colour=(key == "color")),
                })
                records[key].append(record)

    for key, root in targets.items():
        write_metadata(root, records[key])
        write_info(root, args, len(rows), fallbacks, with_colour=(key == "color"))
        make_preview(root, args.variants)
        print(f"wrote {len(records[key])} images to {root}")

    if fallbacks:
        print(f"\nWARNING: {len(fallbacks)} emoji fell back to the 64x64 export: "
              f"{fallbacks[:8]}")


def write_metadata(root, records):
    with open(root / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def write_info(root, args, num_emoji, fallbacks, with_colour):
    # datasets/ is gitignored, so this file plus the script is the only record
    # of how the data was made.
    info = {
        "source": str(SOURCE.relative_to(PROJECT_DIR)),
        "source_image_size": "618x618",
        "image_size": args.image_size,
        "working_size": args.working_size,
        "num_emoji": num_emoji,
        "variants_per_emoji": args.variants,
        "num_images": num_emoji * args.variants,
        "seed": args.seed,
        "colour_augmentation": with_colour,
        "intended_label_col": "group" if with_colour else "name",
        "augmentations": {
            "rotation_deg": [-ROTATION_DEG, ROTATION_DEG],
            "scale_range": list(SCALE_RANGE),
            "hflip": args.hflip,
            **({
                "hue_shift": [-HUE_SHIFT, HUE_SHIFT],
                "saturation_range": list(SATURATION_RANGE),
                "brightness_range": list(BRIGHTNESS_RANGE),
                "contrast_range": list(CONTRAST_RANGE),
            } if with_colour else {}),
        },
        "upsample_fallbacks": fallbacks,
        "license": "CC BY-SA 4.0",
    }
    (root / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def make_preview(root, variants, num_emoji=12, pad=4):
    """Contact sheet: one emoji per row, one variant per column."""
    paths = sorted((root / "images").glob("*.png"))
    if not paths:
        return

    codes = sorted({p.name.rsplit("_a", 1)[0] for p in paths})
    step = max(1, len(codes) // num_emoji)
    codes = codes[::step][:num_emoji]

    size = Image.open(paths[0]).size[0]
    cell = size + pad
    sheet = Image.new(
        "RGB", (cell * variants + pad, cell * len(codes) + pad), (230, 230, 230)
    )
    for r, code in enumerate(codes):
        for c in range(variants):
            tile = root / "images" / f"{code}_a{c:02d}.png"
            if tile.exists():
                sheet.paste(Image.open(tile), (pad + c * cell, pad + r * cell))
    sheet.save(root / "preview.png")


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

def check_colour_safety(size=64):
    """Probe the composite-order invariant directly, on a synthetic glyph.

    The corner check cannot test this on its own, since a zoom-in crop of a
    full-bleed emoji has legitimately coloured corners.  So the invariant that
    actually matters -- colour ops must never tint the white background -- gets
    its own deterministic probe at the extremes of every range.
    """
    rgb = Image.new("RGB", (size, size), WHITE)
    alpha = Image.new("L", (size, size), 0)
    box = (size // 4, size // 4, 3 * size // 4, 3 * size // 4)
    ImageDraw.Draw(rgb).ellipse(box, fill=(200, 60, 40))
    ImageDraw.Draw(alpha).ellipse(box, fill=255)

    problems = []
    for hue, sat, bright, contrast in (
        (HUE_SHIFT, SATURATION_RANGE[1], BRIGHTNESS_RANGE[0], CONTRAST_RANGE[1]),
        (-HUE_SHIFT, SATURATION_RANGE[0], BRIGHTNESS_RANGE[1], CONTRAST_RANGE[0]),
    ):
        params = {"hue": hue, "sat": sat, "bright": bright, "contrast": contrast}
        out = np.asarray(apply_colour(rgb, alpha, params), dtype=np.int16)
        background = out[:size // 8, :size // 8]
        if not (background == 255).all():
            problems.append(
                f"bright={bright} contrast={contrast} tinted the background to "
                f"{background.reshape(-1, 3)[0].tolist()}"
            )
    return problems


def verify(args):
    """Check the contract ConditionedEmojiDataset imposes, without importing it.

    dataloader.py imports torchvision, which is missing from this venv, so the
    checks are reimplemented here in pure PIL/numpy.
    """
    colour_problems = check_colour_safety()
    print("colour ops preserve the white background: "
          + ("ok" if not colour_problems else "FAIL"))
    for problem in colour_problems:
        print(f"  - {problem}")
    ok = not colour_problems
    for suffix in ("geom", "color"):
        root = DATASETS / f"{args.out_prefix}_{suffix}"
        if not root.exists():
            print(f"{root}: MISSING")
            ok = False
            continue

        problems = []
        paths = sorted((root / "images").glob("*.png"))
        with open(root / "metadata.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # The exact condition dataloader.py:85-90 raises on.
        by_file = {row["filename"]: row for row in rows}
        unlabelled = [p.name for p in paths if p.name not in by_file]
        if unlabelled:
            problems.append(f"{len(unlabelled)} image(s) with no CSV row, "
                            f"e.g. {unlabelled[:3]}")
        orphan_rows = [n for n in by_file if not (root / "images" / n).exists()]
        if orphan_rows:
            problems.append(f"{len(orphan_rows)} CSV row(s) with no image, "
                            f"e.g. {orphan_rows[:3]}")
        if len(by_file) != len(rows):
            problems.append(f"duplicate filenames in metadata.csv "
                            f"({len(rows) - len(by_file)} dupes)")

        bad_format, bad_corner = [], []
        for path in paths:
            image = Image.open(path)
            if image.mode != "RGB" or image.size != (args.image_size,) * 2:
                bad_format.append(f"{path.name} {image.mode} {image.size}")
                continue

            # Corners are only guaranteed white where they are pure fill: the
            # untouched original, and zoom-outs whose window extends past the
            # source canvas.  A zoom-in crop of a full-bleed emoji -- sunrise
            # over mountains, night with stars -- lands inside the artwork, so
            # its corners are legitimately coloured.
            match = re.search(r"scale=([\d.]+)", by_file.get(path.name, {}).get("aug_ops", ""))
            if match and float(match.group(1)) < 1.0:
                continue

            w, h = image.size
            corners = [image.getpixel(p)
                       for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
            if any(c != WHITE for c in corners):
                bad_corner.append(f"{path.name} {corners}")
        if bad_format:
            problems.append(f"{len(bad_format)} not 64x64 RGB, e.g. {bad_format[:3]}")
        if bad_corner:
            problems.append(f"{len(bad_corner)} padded/original images with "
                            f"non-white corners, e.g. {bad_corner[:3]}")

        # a00 should reproduce the existing export.
        drifted = []
        for path in paths:
            code, index = path.name.rsplit("_a", 1)
            if index != "00.png":
                continue
            original = SOURCE / "images" / f"{code}.png"
            if not original.exists():
                continue
            a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            b = np.asarray(Image.open(original).convert("RGB"), dtype=np.float32)
            mad = float(np.abs(a - b).mean())
            if mad > 1.0:
                drifted.append(f"{path.name} mad={mad:.2f}")
        if drifted:
            problems.append(f"{len(drifted)} a00 variants differ from the "
                            f"original export, e.g. {drifted[:3]}")

        num_emoji = len({p.name.rsplit("_a", 1)[0] for p in paths})
        expected = num_emoji * args.variants
        if len(paths) != expected:
            problems.append(f"{len(paths)} images, expected {expected}")

        # The label space each dataset is actually meant to be trained on.
        label_col = "group" if suffix == "color" else "name"
        classes = sorted({row[label_col] for row in rows})

        status = "FAIL" if problems else "ok"
        print(f"\n{root.name}: {status}")
        print(f"  {len(paths)} images, {num_emoji} emoji x {args.variants} variants")
        print(f"  {len(classes)} classes on label_col={label_col!r}")
        for problem in problems:
            print(f"  - {problem}")
        ok &= not problems

    print("\nall checks passed" if ok else "\nsome checks FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=int, default=8,
                        help="images per emoji, including the untouched original")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--working-size", type=int, default=256,
                        help="resolution the augmentations are rendered at")
    parser.add_argument("--out-prefix", default="openmoji_aug")
    parser.add_argument("--limit", type=int, help="only the first N emoji")
    parser.add_argument("--hflip", action="store_true",
                        help="allow horizontal flips; off by default because the "
                             "full set contains chirally named emoji such as "
                             "left-/right-facing fist")
    parser.add_argument("--no-download", action="store_true",
                        help="use only what is already in datasets/.cache_618")
    parser.add_argument("--verify", action="store_true",
                        help="check an existing build instead of rebuilding")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.variants < 1:
        raise SystemExit("--variants must be at least 1")
    if args.verify:
        raise SystemExit(verify(args))
    build(args)
    raise SystemExit(verify(args))


if __name__ == "__main__":
    main()
