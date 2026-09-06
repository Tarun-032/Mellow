"""Build the pet sprite sheet from `Pet ideas/` (needs pillow).

    .venv\\Scripts\\python.exe scripts\\sprites.py
    .venv\\Scripts\\python.exe scripts\\sprites.py --check
    .venv\\Scripts\\python.exe scripts\\sprites.py --preview
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ModuleNotFoundError:
    sys.exit("needs pillow:  .venv\\Scripts\\python.exe -m pip install pillow")

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "Pet ideas"
OUT_PNG = ROOT / "src" / "pet" / "sprites.png"
OUT_WRITING = ROOT / "src" / "pet" / "writing.png"
OUT_CSS = ROOT / "src" / "pet" / "sprites.css"
OUT_JSON = ROOT / "src" / "pet" / "sprites.json"
OUT_BUBBLE = ROOT / "src" / "pet" / "bubble.png"
OUT_BONE = ROOT / "src" / "pet" / "bone.png"

# Pointer art: `#` dark, `.` cream, space transparent. Drawn diagonal (no CSS rotate).
BONE = (
    "   ###      ",
    "  #..#      ",
    " #...#      ",
    "#...#       ",
    "#..#.#      ",
    "### #.#     ",
    "     #.# ###",
    "      #.#..#",
    "       #...#",
    "      #...# ",
    "      #..#  ",
    "      ###   ",
)

# Dialogue box size in art pixels (9-slice from `think`).
BUBBLE_ART = (18, 21)  # h, w in art pixels
# 9-slice insets (T R B L); flipped so the tail points down-right.
BUBBLE_SLICE = (5, 8, 7, 6)

CELL = 64  # art pixels per side; pet.css renders this at 3x

# Near-black background threshold (`think` corners peak ~12).
BG_TOL = 20

# Locked five-colour palette (keep in sync with docs/design.md).
PALETTE: dict[str, tuple[int, int, int]] = {
    "cream": (0xF5, 0xEC, 0xDD),
    "tan": (0xC9, 0x82, 0x4A),
    "brown": (0x9B, 0x53, 0x2D),
    "dark": (0x4C, 0x29, 0x23),
    "salmon": (0xF1, 0x89, 0x73),
}

# Sixth colour, for the pencil in `writing`. Deliberately NOT a snap target: as a
# PALETTE entry it would pull Mellow's tan fur toward yellow in every pose. It is
# only ever painted by `writing_details`, which runs after snap_palette.
PENCIL = (0xE8, 0xA8, 0x3C)

# Role-based snap for off-palette generator tones (nearest-colour gets these wrong).
SNAP: dict[tuple[int, int, int], str] = {
    (0xAE, 0x66, 0x40): "tan",  # fur fill in think/yawn; idle uses tan there
    (0xDD, 0xAC, 0x8C): "tan",  # ear crease and muzzle shading, lighter than tan
    (0x87, 0x27, 0x32): "dark",  # inside a yawning mouth, behind the tongue
}

# Large-region recolour for `thinking` (order: brown→tan, then dark→brown).
REPAINT: dict[str, tuple[tuple[str, str], ...]] = {
    "thinking": (("brown", "tan"), ("dark", "brown")),
}
REPAINT_MIN = 20  # px in one connected region

# Flatten legs/feet to cream below these rows.
CREAM_BELOW = {"idle": 59, "thinking": 54, "yawn": 47}

# Max size of isolated dark flecks to repaint from surrounding fur.
DARK_SPECK_MAX = {"petting": 2}

# Ear recolour targets (fill vs crease split by luminance).
EAR_CREAM = PALETTE["cream"]
EAR_TAN = PALETTE["tan"]
EAR_CREASE_LUM = 110
# Min crease pixels before drawing a fold (flattens yawn's lone speck).
CREASE_MIN = 2


@dataclass(frozen=True)
class Pose:
    """One state's sprite cells. Shared `group` = shared crop/placement."""

    file: str | tuple[str, ...]
    group: str
    height: int
    ground: int
    fix_ear: bool = False
    fps: float = 0.0
    # Edge pose: flush right, no shared ground/hit box.
    edge: bool = False
    # Contributes to the shared body/hit union. Off for display-only poses whose
    # props (a notebook, say) would make empty space clickable in every pose.
    union: bool = True

    @property
    def files(self) -> tuple[str, ...]:
        return (self.file,) if isinstance(self.file, str) else self.file


POSES: dict[str, Pose] = {
    "idle": Pose("Mellow main.jpg", "upright", 46, 60),
    "listening": Pose("Mellow Listening.jpg", "upright", 46, 60),
    "talking": Pose("Mellow talking.jpg", "upright", 46, 60),
    "angry": Pose("Mellow Angry.jpg", "upright", 46, 60),
    # Chunkier art; scaled so heads match upright.
    "thinking": Pose("Mellow think.jpg", "think", 42, 60, fix_ear=True),
    "yawn": Pose("Mellow yawn strecth.jpg", "yawn", 46, 60, fix_ear=True),
    "sleeping": Pose("Mellow sleeping.png", "sleep", 26, 60, fix_ear=True),
    # Own crop: different silhouettes from the upright set.
    "petting": Pose("Mellow petting.png", "petting", 46, 60),
    "hunt": Pose("Mellow hunt.png", "hunt", 34, 60),
    # Same 46 as the other standing poses so it does not tower over them. The
    # notebook is part of this silhouette, so the dog reads a little smaller — that
    # is the price of matching footprints, and matching footprints won.
    "writing": Pose(("mellow transcribe notetaking.png",) * 4, "writing", 46, 60, fps=6, union=False),
    # Edge sliver (from peek_art.py); CSS mirrors for the other side.
    "peek": Pose("Mellow peek.png", "peek", 46, 60, edge=True),
}


def label(mask: np.ndarray) -> np.ndarray:
    """8-connected component labels (padded so `np.roll` cannot wrap edges)."""
    m = np.pad(mask, 1)
    lab = np.where(m, np.arange(m.size).reshape(m.shape) + 1, 0)
    shifts = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while True:
        nxt = lab.copy()
        for dy, dx in shifts:
            np.maximum(nxt, np.roll(np.roll(lab, dy, 0), dx, 1), out=nxt)
        nxt[~m] = 0
        if np.array_equal(nxt, lab):
            return lab[1:-1, 1:-1]
        lab = nxt


def largest_blob(mask: np.ndarray, work: int = 384) -> np.ndarray:
    """Keep the largest blob (drops bubble/`?` and watermarks)."""
    h, w = mask.shape
    small = np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((work, work), Image.NEAREST)
    ) > 127

    lab = label(small)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    keep = np.array(Image.fromarray(((lab == ids[counts.argmax()]) * 255).astype(np.uint8))
                    .resize((w, h), Image.NEAREST)) > 127
    return mask & keep


def edge_background(rgb: np.ndarray, work: int = 384) -> np.ndarray:
    """Near-black pixels connected to the frame edge (leaves dark eyes alone)."""
    candidate = rgb.max(2) <= BG_TOL
    h, w = candidate.shape
    small_h = max(1, round(h * work / w))
    small = np.array(
        Image.fromarray(candidate.astype(np.uint8) * 255).resize(
            (work, small_h), Image.NEAREST
        )
    ) > 127
    lab = label(small)
    edge_ids = np.unique(
        np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]))
    )
    exterior = np.isin(lab, edge_ids[edge_ids > 0])
    up = np.array(
        Image.fromarray((exterior * 255).astype(np.uint8)).resize(
            (w, h), Image.NEAREST
        )
    ) > 127
    return candidate & up


def cutout(path: Path) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Load one pose as RGBA with the background transparent, plus its bbox."""
    rgb = np.array(Image.open(path).convert("RGB"))
    mask = largest_blob(~edge_background(rgb))
    rgba = np.dstack([rgb, np.where(mask, 255, 0).astype(np.uint8)])
    ys, xs = np.where(mask)
    return rgba, (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)


def resize_mode(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Mode-resize each block (avoids inventing averaged edge colours)."""
    h, w = img.shape[:2]
    ys = np.linspace(0, h, out_h + 1).round().astype(int)
    xs = np.linspace(0, w, out_w + 1).round().astype(int)

    out = np.zeros((out_h, out_w, 4), np.uint8)
    for r in range(out_h):
        for c in range(out_w):
            block = img[ys[r] : ys[r + 1], xs[c] : xs[c + 1]].reshape(-1, 4)
            colours, counts = np.unique(block, axis=0, return_counts=True)
            out[r, c] = colours[counts.argmax()]
    return out


def brown_blobs(cell: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Fur-brown blobs as (labels, [(id, size, min_x)]); skips near-black outline."""
    rgb = cell[:, :, :3].astype(int)
    brown = (
        (cell[:, :, 3] > 0)
        & (rgb[:, :, 0] - rgb[:, :, 2] > 40)
        & (rgb[:, :, 0] < 230)
        & (rgb.max(2) > 70)
    )
    lab = label(brown)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    found = [
        (int(i), int(n), int(np.where(lab == i)[1].min()))
        for i, n in zip(ids, counts)
        if n >= 20
    ]
    return lab, sorted(found, key=lambda t: t[2])


def recolour_ear(cell: np.ndarray) -> np.ndarray:
    """Recolour the leftmost brown blob (left ear) to cream."""
    lab, blobs = brown_blobs(cell)
    if not blobs:
        return cell
    ear = lab == blobs[0][0]
    lum = cell[:, :, :3].astype(int) @ np.array([0.299, 0.587, 0.114])
    crease = ear & (lum < EAR_CREASE_LUM)
    if crease.sum() < CREASE_MIN:
        crease = np.zeros_like(ear)  # not enough of a fold to be worth drawing
    out = cell.copy()
    out[crease] = (*EAR_TAN, 255)
    out[ear & ~crease] = (*EAR_CREAM, 255)
    return out


def render(poses: dict[str, Pose], art: Path) -> dict[str, list[np.ndarray]]:
    """64x64 RGBA cells per pose, aligned on one ground line."""
    cut: dict[tuple[str, int], tuple[np.ndarray, tuple[int, int, int, int]]] = {}
    for name, p in poses.items():
        for i, f in enumerate(p.files):
            path = art / f
            where = name if len(p.files) == 1 else f"{name} frame {i}"
            if not path.exists():
                print(f"..  skipping {where}: {f} not delivered yet")
                continue
            cut[(name, i)] = cutout(path)

    made: dict[tuple[str, int], np.ndarray] = {}
    for group in dict.fromkeys(poses[n].group for n, _ in cut):
        members = [k for k in cut if poses[k[0]].group == group]
        boxes = np.array([cut[k][1] for k in members])
        y0, x0 = boxes[:, 0].min(), boxes[:, 2].min()
        y1, x1 = boxes[:, 1].max(), boxes[:, 3].max()

        p = poses[members[0][0]]
        out_w = max(1, round(p.height * (x1 - x0) / (y1 - y0)))
        left = CELL - out_w if p.edge else (CELL - out_w) // 2
        top = p.ground - p.height

        for k in members:
            small = resize_mode(cut[k][0][y0:y1, x0:x1], p.height, out_w)
            cell = np.zeros((CELL, CELL, 4), np.uint8)
            # Wider than cell: centre-crop (don't squash).
            sx = max(0, (out_w - CELL) // 2)
            w = min(out_w, CELL)
            cell[top : top + p.height, max(left, 0) : max(left, 0) + w] = small[:, sx : sx + w]
            made[k] = recolour_ear(cell) if poses[k[0]].fix_ear else cell

    return {
        n: [made[(n, i)] for i in range(len(poses[n].files)) if (n, i) in made]
        for n in poses
        if any((n, i) in made for i in range(len(poses[n].files)))
    }


def nth_blob(rgb: np.ndarray, n: int, work: int = 384) -> np.ndarray:
    """Mask of the n-th largest connected region (0 = largest)."""
    mask = rgb.max(2) > BG_TOL
    small = np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((work, work), Image.NEAREST)
    ) > 127
    lab = label(small)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    pick = ids[np.argsort(-counts)[n]]
    up = np.array(
        Image.fromarray(((lab == pick) * 255).astype(np.uint8))
        .resize(rgb.shape[1::-1], Image.NEAREST)
    ) > 127
    return mask & up


def build_bubble(art: Path = ART, out: Path = OUT_BUBBLE, write: bool = True) -> np.ndarray:
    """Lift the dialogue box out of `think` and strip the `?` from inside it."""
    rgb = np.array(Image.open(art / POSES["thinking"].files[0]).convert("RGB"))
    keep = nth_blob(rgb, 1)  # 0 is the dog, 1 is the bubble
    ys, xs = np.where(keep)
    rgba = np.dstack([rgb, np.where(keep, 255, 0).astype(np.uint8)])
    box = resize_mode(rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], *BUBBLE_ART)

    opaque = box[:, :, 3] > 0
    dark = opaque & (box[:, :, :3].max(2) < 110)
    # Strip the `?` (only dark shape not touching an edge).
    lab = label(dark)
    edge = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    fill = Counter(map(tuple, box[opaque & ~dark & (box[:, :, :3].min(2) < 236)][:, :3].tolist()))
    for blob in set(np.unique(lab)) - edge - {0}:
        box[lab == blob] = (*fill.most_common(1)[0][0], 255)

    box = np.ascontiguousarray(box[:, ::-1])  # tail to the right; see BUBBLE_SLICE
    box = snap_palette([box])[0]  # same five colours as the dog

    if write:
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(box).save(out)
        print(f"{out.name}  {BUBBLE_ART[1]}x{BUBBLE_ART[0]} art px, slice {BUBBLE_SLICE}")
    return box


def bone_tip() -> dict[str, int]:
    """Hotspot pixel (Chebyshev-nearest to top-left; derived from BONE)."""
    solid = [
        (y, x)
        for y, row in enumerate(BONE)
        for x, ch in enumerate(row)
        if ch != " "
    ]
    y, x = min(solid, key=lambda p: (max(p), abs(p[0] - p[1])))
    return {"x": x, "y": y}


def make_bone() -> np.ndarray:
    """Rasterise BONE (already on-palette)."""
    box = np.zeros((len(BONE), len(BONE[0]), 4), np.uint8)
    for y, row in enumerate(BONE):
        for x, ch in enumerate(row):
            if ch == " ":
                continue
            box[y, x] = (*PALETTE["dark" if ch == "#" else "cream"], 255)
    return box


def build_bone(out: Path = OUT_BONE, write: bool = True) -> dict:
    """Write bone.png (separate from the strip so it stays out of the hit box)."""
    widths = {len(row) for row in BONE}
    assert len(widths) == 1, f"ragged bone table: {sorted(widths)}"
    height, width = len(BONE), widths.pop()
    box = make_bone()

    if write:
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(box).save(out)
        print(f"{out.name}  {width}x{height} art px, tip {bone_tip()}")
    return {"w": width, "h": height, "tip": bone_tip()}


def snap_palette(cells: list[np.ndarray]) -> list[np.ndarray]:
    """Force every cell onto PALETTE, by role where SNAP says so, else nearest."""
    ref = [(c, n) for n, c in PALETTE.items()] + list(SNAP.items())
    out = []
    for cell in cells:
        opaque = cell[:, :, 3] > 0
        snapped = cell.copy()
        for colour in {tuple(p) for p in cell[opaque][:, :3].tolist()}:
            _, name = min(ref, key=lambda r: sum((a - b) ** 2 for a, b in zip(r[0], colour)))
            snapped[(cell[:, :, :3] == np.array(colour, np.uint8)).all(-1) & opaque, :3] = (
                PALETTE[name]
            )
        out.append(snapped)
    return out


def repaint(name: str, cell: np.ndarray) -> np.ndarray:
    """Apply REPAINT large-region recolours; exit if a rule finds nothing."""
    out = cell.copy()
    for src, dst in REPAINT.get(name, ()):
        hit = (out[:, :, :3] == np.array(PALETTE[src], np.uint8)).all(-1) & (out[:, :, 3] > 0)
        lab = label(hit)
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        big = np.isin(lab, ids[counts >= REPAINT_MIN])
        if not big.any():
            sys.exit(f"{name}: no {src} region >= {REPAINT_MIN}px to repaint - art changed?")
        out[big, :3] = PALETTE[dst]
    return out


def cream_legs(name: str, cell: np.ndarray) -> np.ndarray:
    """Flatten a pose's legs and feet to cream. See CREAM_BELOW."""
    row = CREAM_BELOW.get(name)
    if row is None:
        return cell
    out = cell.copy()
    legs = np.zeros(cell.shape[:2], bool)
    legs[row:] = cell[row:, :, 3] > 0
    out[legs, :3] = PALETTE["cream"]
    return out


def clean_dark_specks(name: str, cell: np.ndarray) -> np.ndarray:
    """Replace tiny isolated dark artifacts with the surrounding fur colour."""
    limit = DARK_SPECK_MAX.get(name)
    if limit is None:
        return cell

    out = cell.copy()
    dark = (
        (out[:, :, :3] == np.array(PALETTE["dark"], np.uint8)).all(-1)
        & (out[:, :, 3] > 0)
    )
    lab = label(dark)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    for blob, count in zip(ids[counts <= limit], counts[counts <= limit]):
        neighbours: list[tuple[int, int, int]] = []
        for y, x in np.argwhere(lab == blob):
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                           (0, 1), (1, -1), (1, 0), (1, 1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < out.shape[0] and 0 <= nx < out.shape[1]):
                    continue
                pixel = out[ny, nx]
                colour = tuple(pixel[:3])
                if pixel[3] > 0 and colour != PALETTE["dark"]:
                    neighbours.append(colour)
        if not neighbours:
            sys.exit(f"{name}: dark speck of {count}px has no fur around it")
        out[lab == blob, :3] = Counter(neighbours).most_common(1)[0][0]
    return out


def body_box(cells: list[np.ndarray]) -> dict[str, int]:
    """Union silhouette bbox for hit-testing / petting."""
    ys, xs = np.where(np.any([c[:, :, 3] > 0 for c in cells], axis=0))
    return {
        "x": int(xs.min()),
        "y": int(ys.min()),
        "w": int(xs.max() - xs.min() + 1),
        "h": int(ys.max() - ys.min() + 1),
    }


def hit_mask(cells: list[np.ndarray]) -> list[str]:
    """Union silhouette used for click-through hit testing in the frontend."""
    opaque = np.any([c[:, :, 3] > 0 for c in cells], axis=0)
    return ["".join("1" if pixel else "0" for pixel in row) for row in opaque]


# Features the 27:1 mode-downscale cannot keep, redrawn at sprite scale.
# Coordinates are read off the rendered cell and pinned by selfcheck(); a change to
# the pose's height, group or source art invalidates them and must fail there.
GLASSES = (
    # (y0, y1, x0, x1) rims, drawn as rounded outlines so the eyes stay visible.
    (26, 33, 18, 26),  # left lens
    (26, 33, 31, 39),  # right lens
)
GLASSES_BRIDGE = (28, 27, 30)  # y, x0, x1
# The downscaled art has its own smudge of a pencil baked into the paw. It has to
# be wiped before a clean one is drawn, or the baked one sits still while the drawn
# one moves — half the pencil stuck, half of it animating.
PENCIL_CLEAR = (37, 49, 14, 23)  # y0, y1, x0, x1 (half-open)
PENCIL_TOP = (38, 16)
PENCIL_TIP = (47, 21)
# Nib offsets per frame: a ping-pong, so the loop reverses instead of snapping back.
NIB_SWING = (0, 1, 0, -1)
# Ruled lines, kept below the paws and clear of the spine.
PAGE_RULES = (
    (52, 15, 23),  # left page
    (54, 14, 22),
    (51, 31, 40),  # right page
    (53, 30, 39),
)


def writing_details(cell: np.ndarray, frame: int) -> np.ndarray:
    """Redraw glasses, pencil and page rules; `frame` swings the nib."""
    out = cell.copy()
    opaque = cell[:, :, 3] > 0

    def put(y, x, colour, only_opaque=True):
        if not (0 <= y < CELL and 0 <= x < CELL):
            return
        if only_opaque and not opaque[y, x]:
            return
        out[y, x] = (*colour, 255)

    for y0, y1, x0, x1 in GLASSES:
        # Corners left off, so the rims read as rounded lenses rather than boxes.
        for x in range(x0 + 1, x1):
            put(y0, x, PALETTE["dark"])
            put(y1, x, PALETTE["dark"])
        for y in range(y0 + 1, y1):
            put(y, x0, PALETTE["dark"])
            put(y, x1, PALETTE["dark"])
    by, bx0, bx1 = GLASSES_BRIDGE
    for x in range(bx0, bx1 + 1):
        put(by, x, PALETTE["dark"])

    # Wipe the baked pencil back to paw/page cream. Only the light pixels go: the
    # notebook spine runs through this box and must survive.
    cy0, cy1, cx0, cx1 = PENCIL_CLEAR
    light = np.zeros(cell.shape[:2], bool)
    for tone in ("cream", "tan", "salmon"):
        light |= (cell[:, :, :3] == np.array(PALETTE[tone], np.uint8)).all(-1)
    band = np.zeros_like(light)
    band[cy0:cy1, cx0:cx1] = True
    out[band & light & opaque, :3] = PALETTE["cream"]

    ty, tx = PENCIL_TOP
    ny, nx = PENCIL_TIP
    nx += NIB_SWING[frame % len(NIB_SWING)]
    steps = max(abs(ny - ty), abs(nx - tx))
    for i in range(steps + 1):
        y = round(ty + (ny - ty) * i / steps)
        x = round(tx + (nx - tx) * i / steps)
        colour = PALETTE["salmon"] if i == 0 else PALETTE["dark"] if i == steps else PENCIL
        put(y, x, colour, only_opaque=False)

    for y, x0, x1 in PAGE_RULES:
        for x in range(x0, x1 + 1):
            put(y, x, PALETTE["tan"])
    return out


def build_writing(art: Path = ART, out: Path = OUT_WRITING):
    """Keep the reference's fine details at the existing 64px logical footprint."""
    pose = POSES["writing"]
    source, (y0, y1, x0, x1) = cutout(art / pose.files[0])
    scale = 4
    width = max(1, round(pose.height * (x1 - x0) / (y1 - y0)))
    left, top = (CELL - width) // 2, pose.ground - pose.height
    small = Image.fromarray(source[y0:y1, x0:x1]).resize(
        (width * scale, pose.height * scale), Image.Resampling.NEAREST,
    )
    base = np.zeros((CELL * scale, CELL * scale, 4), np.uint8)
    base[top*scale:pose.ground*scale, left*scale:(left+width)*scale] = np.array(small)
    frames = []
    cy0, cy1, cx0, cx1 = (value * scale for value in PENCIL_CLEAR)
    yy, xx = np.mgrid[cy0:cy1, cx0:cx1]
    # Keep the existing nib swing, with a feathered displacement of the original art.
    weight = np.sin(np.pi * (xx-cx0) / (cx1-cx0-1)) * np.sin(np.pi * (yy-cy0) / (cy1-cy0-1))
    for frame in range(len(pose.files)):
        cell = base.copy()
        shift = np.rint(weight * NIB_SWING[frame % len(NIB_SWING)] * scale / 2).astype(int)
        cell[cy0:cy1, cx0:cx1] = base[yy, xx-shift]
        frames.append(cell)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(frames, axis=1)).save(out)
    print(f"{out.name}: {len(frames)} frames; unchanged {CELL}x{CELL} logical cell, height {pose.height}, ground {pose.ground}")


def build(art: Path = ART, out_png: Path = OUT_PNG, write: bool = True):
    frames = render(POSES, art)
    if not frames:
        sys.exit(f"no art found in {art}")

    # Same corrections for every frame of a pose.
    keys = [(n, i) for n in frames for i in range(len(frames[n]))]
    snapped = snap_palette([frames[n][i] for n, i in keys])
    flat = [
        clean_dark_specks(n, cream_legs(n, repaint(n, c)))
        for (n, _), c in zip(keys, snapped)
    ]
    for index, (name, frame) in enumerate(keys):
        if name == "writing":
            flat[index] = writing_details(flat[index], frame)

    cells: dict[str, list[np.ndarray]] = {}
    for (n, _), c in zip(keys, flat):
        cells.setdefault(n, []).append(c)

    # Start index and frame count per pose, in sheet order.
    index, at = {}, 0
    for n, fs in cells.items():
        index[n] = {"cell": at, "frames": len(fs), "fps": POSES[n].fps}
        at += len(fs)

    standing = [c for (n, _), c in zip(keys, flat) if not POSES[n].edge and POSES[n].union]

    if write:
        build_writing(art)
        sheet = np.concatenate(flat, axis=1)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sheet).save(out_png)

        meta = {
            "cell": CELL,
            # Shared ground row for motion pivots.
            "ground": POSES[next(iter(cells))].ground,
            "body": body_box(standing),
            "hit": hit_mask(standing),
            # Per-edge-pose body box (not the shared standing union).
            **{n: body_box(cells[n]) for n in cells if POSES[n].edge},
            # Bone meta (not in the strip); tip for the frontend.
            "bone": build_bone(write=write),
            "frames": index,
        }
        OUT_JSON.write_text(json.dumps(meta, indent=2) + "\n")

        # Generated CSS vars so pose indices stay in sync.
        lines = ["/* Generated by scripts/sprites.py - do not edit. */", ":root {"]
        for n, f in index.items():
            lines.append(f"  --cell-{n}: {f['cell']};")
            lines.append(f"  --frames-{n}: {f['frames']};")
            lines.append(f"  --fps-{n}: {f['fps']:g};")
            seconds = f["frames"] / f["fps"] if f["frames"] > 1 and f["fps"] else 1
            play = "running" if f["frames"] > 1 and f["fps"] else "paused"
            lines.append(f"  --frame-time-{n}: {seconds:g}s;")
            lines.append(f"  --frame-play-{n}: {play};")
        lines += [
            f"  --cell-count: {at};",
            f"  --ground: {meta['ground']};",
            f"  --body-x: {meta['body']['x']};",
            f"  --body-y: {meta['body']['y']};",
            f"  --body-w: {meta['body']['w']};",
            f"  --body-h: {meta['body']['h']};",
            f"  --bone-w: {meta['bone']['w']};",
            f"  --bone-h: {meta['bone']['h']};",
            f"  --bone-tip-x: {meta['bone']['tip']['x']};",
            f"  --bone-tip-y: {meta['bone']['tip']['y']};",
            "}",
            "",
        ]
        OUT_CSS.write_text("\n".join(lines))

    shape = ", ".join(f"{n}x{len(fs)}" if len(fs) > 1 else n for n, fs in cells.items())
    print(f"{at} cells over {len(cells)} poses, {len(PALETTE)} colours + pencil: {shape}")
    return cells


def preview(cells: dict[str, list[np.ndarray]], path: Path, zoom: int = 6) -> None:
    """Big side-by-side PNG with the ground line drawn on, for tuning POSES."""
    sheet = np.concatenate([c for fs in cells.values() for c in fs], axis=1)
    img = np.dstack([np.full(sheet.shape[:2], 24, np.uint8)] * 3)
    a = sheet[:, :, 3:4] / 255
    img = (sheet[:, :, :3] * a + img * (1 - a)).astype(np.uint8)
    img[POSES["idle"].ground] = [255, 0, 128]  # the line every dog must stand on
    big = Image.fromarray(img).resize(
        (sheet.shape[1] * zoom, sheet.shape[0] * zoom), Image.NEAREST
    )
    big.save(path)
    print(f"preview -> {path}")


def selfcheck() -> None:
    """Assert shared ground/centre and palette invariants."""
    poses = build(write=False)

    # One entry per cell for the assertions below.
    cells: dict[str, np.ndarray] = {}
    pose_of: dict[str, str] = {}
    for n, fs in poses.items():
        for i, c in enumerate(fs):
            lab = n if len(fs) == 1 else f"{n}[{i}]"
            cells[lab], pose_of[lab] = c, n

    # Partial frame delivery is a silent animation bug; catch it here.
    for n, fs in poses.items():
        assert len(fs) == len(POSES[n].files), (
            f"{n}: {len(fs)} of {len(POSES[n].files)} frames delivered"
        )

    for n, c in cells.items():
        assert c.shape == (CELL, CELL, 4), (n, c.shape)
        assert c[0, 0, 3] == 0 and c[-1, -1, 3] == 0, f"{n}: corners not keyed"
        assert (c[:, :, 3] > 0).any(), f"{n}: cell is empty"
    print(f"ok  {len(cells)} cells are {CELL}x{CELL}, keyed and non-empty")

    # Skip edge poses for shared ground/centre checks.
    feet, mids = {}, {}
    for n, c in cells.items():
        if POSES[n].edge:
            continue
        ys, xs = np.where(c[:, :, 3] > 0)
        feet[n], mids[n] = ys.max(), (xs.min() + xs.max()) / 2
    assert max(feet.values()) - min(feet.values()) <= 2, f"ground lines differ: {feet}"
    assert max(mids.values()) - min(mids.values()) <= 3, f"centres differ: {mids}"
    print(f"ok  every dog stands on the same ground line (row {max(feet.values())})")

    # Upright group: body silhouette must match; only the head may differ.
    upright = [n for n in cells if POSES[pose_of[n]].group == "upright"]
    if len(upright) > 1:
        base = cells[upright[0]]
        first = POSES[pose_of[upright[0]]]
        shoulder = first.ground - first.height // 2
        for n in upright[1:]:
            body = base[shoulder:, :, 3] > 0
            moved = (cells[n][shoulder:, :, 3] > 0) != body
            assert moved.sum() / body.sum() < 0.05, (
                f"{n}: body outline differs from {upright[0]} by "
                f"{moved.sum()}px below the shoulder"
            )
        print(f"ok  {', '.join(upright)} share one body outline — only the head moves")

    # One connected shape per cell (catches leftover bubble/watermark).
    for n, c in cells.items():
        lab = label(c[:, :, 3] > 0)
        assert len(np.unique(lab)) == 2, f"{n}: {len(np.unique(lab)) - 1} shapes, expected 1"
    print("ok  every cell is a single shape — `?` bubble and watermark stripped")

    # Hunt eyes must stay opaque under the moving highlights.
    if "hunt" in cells:
        hunt = cells["hunt"]
        assert hunt[48:52, 19:23, 3].all(), "hunt: left eye contains transparency"
        assert hunt[47:52, 33:38, 3].all(), "hunt: right eye contains transparency"
        print("ok  hunt eyes are fully opaque beneath the moving highlights")

    # Automated ear-cream check for fix_ear poses only.
    fixed = [n for n in cells if POSES[pose_of[n]].fix_ear]
    for n in fixed:
        left = cells[n][:, : CELL // 3]
        assert (left[:, :, :3] == np.array(EAR_CREAM, np.uint8)).all(-1).any(), (
            f"{n}: ear recolour found nothing to change"
        )
    print(f"ok  ear recoloured cream on {', '.join(fixed)}")

    # Feet in the last rows must be cream after CREAM_BELOW.
    creamed = [n for n in cells if pose_of[n] in CREAM_BELOW]
    for n in creamed:
        feet = cells[n][-4:]
        paw = feet[feet[:, :, 3] > 0][:, :3]
        assert (paw == np.array(PALETTE["cream"], np.uint8)).all(), f"{n}: feet aren't cream"
    print(f"ok  all four feet are cream on {', '.join(creamed)}")

    seen = {tuple(p) for c in cells.values() for p in c[c[:, :, 3] > 0][:, :3].tolist()}
    extra = seen - set(PALETTE.values())
    assert not extra, "off-palette: " + ", ".join("#%02x%02x%02x" % c for c in sorted(extra))
    print(f"ok  every pixel is one of the {len(PALETTE)} locked colours")

    # Bubble and bone (not in the strip) must also be on-palette.
    bone = make_bone()
    for name, cell in (("bubble", build_bubble(write=False)), ("bone", bone)):
        assert cell.shape[2] == 4, f"{name}: no alpha"
        assert (cell[:, :, 3] > 0).any(), f"{name}: nothing drawn"
        shades = {tuple(p) for p in cell[cell[:, :, 3] > 0][:, :3].tolist()}
        loose = shades - set(PALETTE.values())
        assert not loose, f"{name} off-palette: " + ", ".join(
            "#%02x%02x%02x" % c for c in sorted(loose)
        )
    assert bone.shape[:2] == (len(BONE), len(BONE[0])), (bone.shape, len(BONE))
    tip = bone_tip()
    assert bone[tip["y"], tip["x"], 3] > 0, f"the hotspot {tip} is a transparent pixel"
    print(
        f"ok  bubble and bone are on-palette; bone is"
        f" {len(BONE[0])}x{len(BONE)} with its tip at {tip['x']},{tip['y']}"
    )
    print("\nall checks passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        selfcheck()
    elif "--preview" in sys.argv:
        preview(build(write=False), ROOT / "sprites-preview.png")
    else:
        build()
        build_bubble()
        build_bone()
