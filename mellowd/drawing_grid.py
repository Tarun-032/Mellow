"""Bounded, capture-local calibration of a visible row of repeated cells.

The model proposes a row and its numbering. Geometry comes from independently
measured control bounds or native source pixels. This module does not recognise
an application, execute clicks, infer hidden cells, or relocate an old grid.
"""

from dataclasses import dataclass
import math

import numpy as np

MIN_CELLS = 4
MAX_CELLS = 32
MAX_INDEX = 512
MAX_AREA = 400_000
CONTRAST = 12


@dataclass(frozen=True)
class Cell:
    index: int
    bounds: tuple

    @property
    def center(self):
        x, y, width, height = self.bounds
        return (x + width / 2, y + height / 2)


@dataclass(frozen=True)
class Calibration:
    capture_id: str
    region: tuple
    cells: tuple
    method: str
    # Pixel regularity proves geometry, never the model's row name or step
    # numbering. The caller must establish any semantic claim independently.
    semantic_index_verified: bool = False

    def cell(self, index):
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("grid index must be an integer")
        return next((cell for cell in self.cells if cell.index == index), None)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("grid coordinates must be finite numbers")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError("grid coordinates must be finite numbers") from None
    if not math.isfinite(value):
        raise ValueError("grid coordinates must be finite numbers")
    return value


def _rect(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("grid region needs physical x, y, width and height")
    x, y, width, height = map(_number, value)
    if min(width, height) <= 0:
        raise ValueError("grid region is empty")
    return x, y, width, height


def _enclosure(boxes):
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return left, top, right - left, bottom - top


def _checked(boxes, proposed, count):
    """Verify the actual boxes; never interpolate an unobserved cell."""
    if len(boxes) < 2:
        # No row structure at all (custom-drawn or low-contrast cells), as
        # opposed to a structure that disagrees with the proposed count.
        raise ValueError("grid cells are not visible as separate boxes")
    if len(boxes) != count:
        raise ValueError("grid cell count does not match observed geometry")
    boxes = sorted((_rect(box) for box in boxes), key=lambda box: box[0])
    widths = np.array([box[2] for box in boxes])
    heights = np.array([box[3] for box in boxes])
    tops = np.array([box[1] for box in boxes])
    centers = np.array([box[0] + box[2] / 2 for box in boxes])
    pitch = np.diff(centers)
    median_pitch = float(np.median(pitch))
    if min(widths.min(), heights.min()) < 4 or median_pitch < 5:
        raise ValueError("grid cells are too small to calibrate")
    if (np.ptp(widths) > max(1.5, float(np.median(widths)) * .08)
            or np.ptp(heights) > max(1.5, float(np.median(heights)) * .08)
            or np.ptp(tops) > max(1.5, float(np.median(heights)) * .05)):
        raise ValueError("grid cells do not form one aligned row")
    if np.max(np.abs(pitch - median_pitch)) > max(1.5, median_pitch * .055):
        raise ValueError("grid cell spacing is irregular")
    gaps = np.array([b[0] - a[0] - a[2] for a, b in zip(boxes, boxes[1:])])
    if gaps.min() < 1 or np.ptp(gaps) > max(1.5, float(np.median(gaps)) * .20):
        raise ValueError("grid cells overlap or have irregular gaps")
    observed = _enclosure(boxes)
    px, py, pw, ph = proposed
    ox, oy, ow, oh = observed
    tolerance = max(2, min(6, median_pitch * .15))
    # A proposal must enclose the same complete row, rather than a nearby row,
    # a clipped subset, its label, or a larger collection of controls.
    if (abs(ox - px) > tolerance or abs(ox + ow - px - pw) > tolerance
            or abs(oy - py) > tolerance or abs(oy + oh - py - ph) > tolerance):
        raise ValueError("observed grid does not match the proposed row")
    return tuple(boxes), observed


def _components(mask):
    """Four-connected foreground components using bounded horizontal runs."""
    parents, bounds, sizes = [], [], []

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    previous = []
    for y, row in enumerate(mask):
        padded = np.pad(row, 1).astype(np.int8)
        changes = np.diff(padded)
        runs = zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
        current = []
        cursor = 0
        for left, right in runs:
            index = len(parents)
            parents.append(index)
            bounds.append((int(left), y, int(right), y + 1))
            sizes.append(int(right - left))
            while cursor < len(previous) and previous[cursor][1] <= left:
                cursor += 1
            neighbor = cursor
            while neighbor < len(previous) and previous[neighbor][0] < right:
                parents[root(previous[neighbor][2])] = root(index)
                neighbor += 1
            current.append((int(left), int(right), index))
        previous = current
    groups = {}
    for index, (left, top, right, bottom) in enumerate(bounds):
        key = root(index)
        if key not in groups:
            groups[key] = [left, top, right, bottom, sizes[index]]
        else:
            item = groups[key]
            item[0] = min(item[0], left)
            item[1] = min(item[1], top)
            item[2] = max(item[2], right)
            item[3] = max(item[3], bottom)
            item[4] += sizes[index]
    return tuple(groups.values())


def _sample(rgb, monitor, proposed, count):
    x, y, width, height = proposed
    pitch = width / count
    padding = max(2, min(8, round(pitch * .16)))
    left = math.floor(x - monitor["left"]) - padding
    top = math.floor(y - monitor["top"]) - padding
    right = math.ceil(x + width - monitor["left"]) + padding
    bottom = math.ceil(y + height - monitor["top"]) + padding
    if left < 0 or top < 0 or right > rgb.shape[1] or bottom > rgb.shape[0]:
        raise ValueError("grid border is clipped by the monitor")
    image = rgb[top:bottom, left:right].astype(np.int16)
    collar = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    background = np.median(collar, axis=0)
    if np.mean(np.max(np.abs(collar - background), axis=1) < CONTRAST) < .75:
        raise ValueError("grid row has no unambiguous background boundary")
    return image, background, left, top


def _pixel_boxes(rgb, monitor, proposed, count):
    image, background, left, top = _sample(rgb, monitor, proposed, count)
    _, _, width, height = proposed
    pitch = width / count
    # RGB contrast retains equal-luminance state colours. Projecting each
    # connected component's borders verifies actual rectangular cell runs;
    # alternating on/off fills cannot silently change the inferred pitch.
    mask = np.max(np.abs(image - background), axis=2) >= CONTRAST
    boxes = []
    for l, t, r, b, size in _components(mask):
        w, h = r - l, b - t
        if w < max(4, pitch * .40) or h < max(4, height * .45) or size < 8:
            continue  # Tiny interior glyphs are not a visible cell boundary.
        if l == 0 or t == 0 or r == image.shape[1] or b == image.shape[0]:
            raise ValueError("grid cell is clipped or continues outside its row")
        area = mask[t:b, l:r]
        if min(area[0].mean(), area[-1].mean(), area[:, 0].mean(), area[:, -1].mean()) < .75:
            raise ValueError("grid cell has no complete rectangular boundary")
        boxes.append((monitor["left"] + left + l, monitor["top"] + top + t, w, h))
    return boxes, background


def _adjacent_cell(rgb, monitor, boxes, background):
    """A neighbouring observed rectangle disproves a whole-row proposal.

    Inspect native pixels one pitch beyond each end. This detects evidence of
    continuation, not a template-based relocation or inferred step numbering.
    """
    first, last = boxes[0], boxes[-1]
    pitch = float(np.median([b[0] - a[0] for a, b in zip(boxes, boxes[1:])]))
    width = float(np.median([box[2] for box in boxes]))
    height = float(np.median([box[3] for box in boxes]))
    expected = ((first[0] - pitch, first[1]), (last[0] + pitch, last[1]))
    tolerance = max(2, min(4, pitch * .12))
    for x, y in expected:
        left = max(0, math.floor(x - monitor["left"] - 2))
        right = min(rgb.shape[1], math.ceil(x + width - monitor["left"] + 2))
        top = max(0, math.floor(y - monitor["top"] - 2))
        bottom = min(rgb.shape[0], math.ceil(y + height - monitor["top"] + 2))
        if left >= right or top >= bottom:
            continue
        area = rgb[top:bottom, left:right].astype(np.int16)
        mask = np.max(np.abs(area - background), axis=2) >= CONTRAST
        for l, t, r, b, _ in _components(mask):
            bx, by = monitor["left"] + left + l, monitor["top"] + top + t
            if (abs(bx - x) <= tolerance and abs(by - y) <= tolerance
                    and abs(r - l - width) <= tolerance
                    and abs(b - t - height) <= tolerance):
                boundary = mask[t:b, l:r]
                if min(boundary[0].mean(), boundary[-1].mean(),
                       boundary[:, 0].mean(), boundary[:, -1].mean()) >= .75:
                    return True
    return False


def calibrate(pixels, monitor, region, *, count, first_index=1, capture_id,
              measured_bounds=()):
    """Return observed physical cells for this capture, or fail closed.

    ``region`` is physical desktop ``(left, top, width, height)``. The input
    array must be the full native-resolution monitor, not the uploaded image.
    ``measured_bounds`` may contain ONLY independently measured UIA cells from
    this source capture, selected by the caller; model rectangles are not UIA.
    ``first_index`` is a semantic proposal, never a pixel-verified step label.
    No calibration is cached or reusable across captures.
    """
    if (isinstance(count, bool) or not isinstance(count, int)
            or not MIN_CELLS <= count <= MAX_CELLS):
        raise ValueError("grid needs four to thirty-two observed cells")
    if (isinstance(first_index, bool) or not isinstance(first_index, int)
            or not 1 <= first_index <= MAX_INDEX - count + 1):
        raise ValueError("grid first index is invalid")
    if not isinstance(capture_id, str) or not capture_id or len(capture_id) > 128:
        raise ValueError("grid needs a source capture identity")
    if not isinstance(monitor, dict):
        raise ValueError("grid needs native monitor geometry")
    try:
        mx, my, mw, mh = map(_number, (monitor["left"], monitor["top"],
                                      monitor["width"], monitor["height"]))
    except KeyError:
        raise ValueError("grid needs native monitor geometry") from None
    if min(mw, mh) <= 0 or mw != int(mw) or mh != int(mh):
        raise ValueError("grid needs native monitor geometry")
    if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
            or pixels.ndim not in (2, 3) or pixels.shape[:2] != (int(mh), int(mw))
            or (pixels.ndim == 3 and pixels.shape[2] not in (3, 4))):
        raise ValueError("grid pixels must match the native monitor")
    proposed = _rect(region)
    x, y, width, height = proposed
    if (x < mx or y < my or x + width > mx + mw or y + height > my + mh
            or width * height > MAX_AREA or width > 8192 or height > 512):
        raise ValueError("grid row is outside the bounded source region")
    measured_bounds = tuple(measured_bounds)
    rgb = np.repeat(pixels[:, :, None], 3, axis=2) if pixels.ndim == 2 else pixels[:, :, :3]
    background = None
    if measured_bounds:
        boxes = measured_bounds
        method = "uia"
    else:
        boxes, background = _pixel_boxes(rgb, monitor, proposed, count)
        method = "pixels"
    boxes, observed = _checked(boxes, proposed, count)
    if any(bx < mx or by < my or bx + bw > mx + mw or by + bh > my + mh
           for bx, by, bw, bh in boxes):
        raise ValueError("grid cell is outside the source monitor")
    if method == "uia":
        try:
            _, background, _, _ = _sample(rgb, monitor, observed, count)
        except ValueError:
            # Independent UIA geometry remains usable on a low-contrast or
            # textured source. Pixel inability is not evidence of continuation.
            pass
    if background is not None and _adjacent_cell(rgb, monitor, boxes, background):
        raise ValueError("grid proposal is a subset of a longer visible row")
    cells = tuple(Cell(first_index + index, box) for index, box in enumerate(boxes))
    return Calibration(capture_id, observed, cells, method)
