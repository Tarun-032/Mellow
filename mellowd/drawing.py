"""Local annotation geometry. No model calls and no executable drawing input."""

from dataclasses import dataclass, field
import math
import time
import uuid

import numpy as np
from PIL import Image

MAX_MARKS = 32
MAX_VERTICES = 16
MAX_LABEL = 60
LIFETIME = 12.0
# Freshness of a captured plan is separate from the time allowed to explain it.
ACTIVE_LIFETIME = 60.0
COLORS = {"mint", "amber", "violet"}
COUNTS = {"rectangle": 2, "highlight": 2, "ellipse": 2, "line": 2,
          "arrow": 2, "label": 1, "quadratic": 3, "cubic": 4}


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("drawing coordinates must be finite numbers")
    try:
        finite = float(value)
    except OverflowError:
        raise ValueError("drawing coordinates must be finite numbers") from None
    if not math.isfinite(finite):
        raise ValueError("drawing coordinates must be finite numbers")
    return finite


@dataclass(frozen=True)
class Transform:
    """Continuous image edges: 0 maps to the left edge, 1000 to the right.

    Cropping precedes resize and padding. Coordinates in padding are rejected.
    Host geometry is always physical desktop pixels, never pointer fractions.
    """

    source_left: float
    source_top: float
    crop_left: float
    crop_top: float
    crop_width: float
    crop_height: float
    image_width: int
    image_height: int
    resize_x: float
    resize_y: float
    padding_x: float = 0
    padding_y: float = 0

    def point(self, value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("a drawing point needs two coordinates")
        x, y = map(number, value)
        if not 0 <= x <= 1000 or not 0 <= y <= 1000:
            raise ValueError("drawing point is outside its image")
        fields = tuple(self.__dict__.values())
        if any(not math.isfinite(number(v)) for v in fields) or min(
            self.crop_width, self.crop_height, self.image_width, self.image_height,
            self.resize_x, self.resize_y
        ) <= 0 or min(self.crop_left, self.crop_top, self.padding_x, self.padding_y) < 0:
            raise ValueError("invalid image transform")
        if (self.padding_x + self.crop_width * self.resize_x > self.image_width + 1e-6
                or self.padding_y + self.crop_height * self.resize_y > self.image_height + 1e-6):
            raise ValueError("transformed crop does not fit its image")
        x = x * self.image_width / 1000 - self.padding_x
        y = y * self.image_height / 1000 - self.padding_y
        if not 0 <= x <= self.crop_width * self.resize_x or not 0 <= y <= self.crop_height * self.resize_y:
            raise ValueError("drawing point lies in image padding")
        return [self.source_left + self.crop_left + x / self.resize_x,
                self.source_top + self.crop_top + y / self.resize_y]


def thumbnail(pixels):
    return np.asarray(Image.fromarray(pixels).convert("L").resize((240, 135)), dtype=np.int16)


def observation(pixels):
    """Enough local detail to notice a changed control, without keeping RGB."""
    return np.asarray(Image.fromarray(pixels).convert("L").resize((480, 270)), dtype=np.int16)


def _edges(gray):
    edge = np.zeros(gray.shape, dtype=bool)
    horizontal = np.abs(gray[:, 1:] - gray[:, :-1]) > 22
    vertical = np.abs(gray[1:, :] - gray[:-1, :]) > 22
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return edge


def _nearby(edge):
    """One sample pixel of antialias/resampling tolerance for broad layout."""
    padded = np.pad(edge, 1)
    nearby = np.zeros_like(edge)
    for y in range(3):
        for x in range(3):
            nearby |= padded[y:y + edge.shape[0], x:x + edge.shape[1]]
    return nearby


def _reversed_edges(before, after):
    """Text/state changes reverse gradients; a uniform hover fill does not."""
    changed = np.zeros(before.shape, dtype=bool)
    for axis in (0, 1):
        old, new = np.diff(before, axis=axis), np.diff(after, axis=axis)
        reverse = (((old < 0) & (new > 0)) | ((old > 0) & (new < 0)))
        reverse &= np.minimum(np.abs(old), np.abs(new)) > 12
        if axis:
            changed[:, 1:] |= reverse
            changed[:, :-1] |= reverse
        else:
            changed[1:, :] |= reverse
            changed[:-1, :] |= reverse
    return changed


def _layout_replaced(edges):
    """Detect coherent local layout changes, even on mostly empty screens.

    Four-sample tiles join the nearby edges of one changed object. Narrow
    status text/caret changes stay small; a relocated block/dialog does not.
    """
    height, width = edges.shape
    padded = np.pad(edges, ((0, (-height) % 4), (0, (-width) % 4)))
    tiles = padded.reshape(padded.shape[0] // 4, 4, padded.shape[1] // 4, 4).any(axis=(1, 3))
    seen = set()
    for y, x in zip(*np.nonzero(tiles)):
        first = (int(y), int(x))
        if first in seen:
            continue
        seen.add(first)
        pending, cells = [first], []
        while pending:
            row, column = pending.pop()
            cells.append((row, column))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (row + dy, column + dx)
                    if (neighbor not in seen and 0 <= neighbor[0] < tiles.shape[0]
                            and 0 <= neighbor[1] < tiles.shape[1] and tiles[neighbor]):
                        seen.add(neighbor)
                        pending.append(neighbor)
        top, bottom = min(y for y, _ in cells) * 4, (max(y for y, _ in cells) + 1) * 4
        left, right = min(x for _, x in cells) * 4, (max(x for _, x in cells) + 1) * 4
        if right - left >= 32 and bottom - top >= 32 and edges[top:bottom, left:right].sum() >= 80:
            return True
    return False


def _region(rect, monitor, shape):
    """A measured physical rectangle clipped to a small local observation."""
    try:
        x, y, width, height = map(number, rect)
        if width <= 0 or height <= 0:
            return None
    except (TypeError, ValueError):
        return None
    rows, columns = shape
    left = max(0, min(columns, math.floor((x - monitor["left"]) / monitor["width"] * columns)))
    top = max(0, min(rows, math.floor((y - monitor["top"]) / monitor["height"] * rows)))
    right = max(0, min(columns, math.ceil((x + width - monitor["left"]) / monitor["width"] * columns)))
    bottom = max(0, min(rows, math.ceil((y + height - monitor["top"]) / monitor["height"] * rows)))
    return (slice(top, bottom), slice(left, right)) if right > left and bottom > top else None


def _animation_stable(before, after, region):
    """An observed animation may rearrange pixels inside its fixed image frame.

    The frame and its immediate surrounding source stay observable. A moved,
    removed, recolored or replaced canvas fails independently of its interior
    motion. This exception is for a whole-image enclosure, never precise paths
    or a claimed action result inside an animated image.
    """
    ys, xs = region
    old, new = before[region], after[region]
    height, width = old.shape
    if min(height, width) < 16:
        return False
    # Two observation samples around the frame include caption/background
    # anchors. Preserve the border itself as well, not an arbitrary mark box.
    top, left = max(0, ys.start - 2), max(0, xs.start - 2)
    bottom, right = min(before.shape[0], ys.stop + 2), min(before.shape[1], xs.stop + 2)
    collar = np.ones((bottom - top, right - left), dtype=bool)
    collar[ys.start - top + 2:ys.stop - top - 2,
           xs.start - left + 2:xs.stop - left - 2] = False
    boundary_change = np.abs(after[top:bottom, left:right] - before[top:bottom, left:right]) > 18
    if (boundary_change & collar).sum() > max(8, collar.sum() * .04):
        return False
    # Moving colored triangles preserve the canvas' intensity distribution;
    # blanking it or showing a different panel does not. Broad histogram bins
    # tolerate antialiasing while retaining a check on every interior pixel.
    old_hist = np.bincount((old.ravel() // 16).clip(0, 15), minlength=16) / old.size
    new_hist = np.bincount((new.ravel() // 16).clip(0, 15), minlength=16) / new.size
    return bool(np.max(np.abs(np.cumsum(old_hist - new_hist))) < .10)


@dataclass(frozen=True)
class Frame:
    monitor: dict
    hwnd: int
    window: tuple | None
    transform: Transform
    fingerprint: object = field(repr=False)
    captured: float = field(default_factory=time.monotonic)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    content_fingerprint: object = field(default=None, repr=False)

    @classmethod
    def capture(cls, monitor, hwnd, window, width, height, pixels, *, captured=None):
        return cls(dict(monitor), hwnd, window, Transform(
            monitor["left"], monitor["top"], 0, 0, monitor["width"], monitor["height"],
            width, height, width / monitor["width"], height / monitor["height"]
        ), thumbnail(pixels), captured if captured is not None else time.monotonic(),
                   content_fingerprint=observation(pixels))

    def scene(self, marks, reveal_from=0):
        if not isinstance(marks, list) or not 1 <= len(marks) <= MAX_MARKS:
            raise ValueError("a drawing scene needs 1–32 marks")
        if type(reveal_from) is not int or not 0 <= reveal_from <= len(marks):
            raise ValueError("invalid reveal boundary")
        if time.monotonic() - self.captured > LIFETIME:
            raise ValueError("drawing frame has expired")
        out = []
        for mark in marks:
            if not isinstance(mark, dict) or set(mark) - {"kind", "points", "text", "color"}:
                raise ValueError("unsupported drawing fields")
            kind = mark.get("kind")
            points = mark.get("points")
            if not isinstance(kind, str) or not isinstance(points, list) or (
                kind == "polygon" and not 3 <= len(points) <= MAX_VERTICES
            ) or (kind != "polygon" and (kind not in COUNTS or len(points) != COUNTS[kind])):
                raise ValueError("invalid drawing kind or vertex count")
            color = mark.get("color", "mint")
            text = mark.get("text")
            if not isinstance(color, str) or color not in COLORS:
                raise ValueError("unsupported drawing color")
            if kind == "label":
                if not isinstance(text, str) or not 1 <= len(text) <= MAX_LABEL or any(ord(c) < 32 for c in text):
                    raise ValueError("invalid drawing label")
            elif text is not None:
                raise ValueError("only labels may carry text")
            physical = [self.transform.point(p) for p in points]
            mon = self.monitor
            if any(not mon["left"] <= x <= mon["left"] + mon["width"] or
                   not mon["top"] <= y <= mon["top"] + mon["height"] for x, y in physical):
                raise ValueError("drawing leaves its source monitor")
            if kind in {"rectangle", "highlight", "ellipse"} and (
                physical[0][0] >= physical[1][0] or physical[0][1] >= physical[1][1]
            ):
                raise ValueError("drawing box must have positive size")
            out.append(dict(kind=kind, points=physical, text=text, color=color))
        return dict(frame_id=self.id, hwnd=self.hwnd, window=list(self.window) if self.window else None,
                    monitor=self.monitor, lifetime_ms=int(ACTIVE_LIFETIME * 1000),
                    marks=out, reveal_from=reveal_from)

    def animation_regions(self, samples, image_rects):
        """Recognize ongoing motion in small measured source-image containers.

        At least two later clean samples must actually differ; an image name
        or a one-time replacement grants no exception. Source movement outside
        the containers invalidates the observation. These rectangles are local
        ephemeral evidence; never instructions or persisted user data.
        """
        baseline = self.content_fingerprint
        if baseline is None or len(samples) < 2:
            return ()
        try:
            observed = [np.asarray(Image.fromarray(p).convert("L").resize(
                (baseline.shape[1], baseline.shape[0])), dtype=np.int16) for p in samples[-8:]]
        except (AttributeError, TypeError, ValueError):
            return ()
        source = _region(self.window or (self.monitor["left"], self.monitor["top"],
                                        self.monitor["width"], self.monitor["height"]),
                         self.monitor, baseline.shape)
        if source is None:
            return ()
        source_area = baseline[source].size
        approved, occupied = [], 0
        for rect in list(image_rects)[:16]:
            region = _region(rect, self.monitor, baseline.shape)
            if region is None:
                continue
            area = baseline[region].size
            if area > source_area * .20 or occupied + area > source_area * .25:
                continue
            if not all(_animation_stable(baseline, sample, region) for sample in observed):
                continue
            # Multiple changed phases distinguish ongoing motion from a single
            # image swap. Tiny compression/hover differences do not qualify.
            phases = [baseline, *observed]
            changes = [int((np.abs(a[region] - b[region]) > 18).sum())
                       for a, b in zip(phases, phases[1:])]
            if sum(n >= max(12, area * .005) for n in changes) < 2:
                continue
            approved.append(tuple(map(number, rect)))
            occupied += area
        if not approved:
            return ()
        # A local animation cannot excuse scroll/navigation of the page around
        # it, even when the candidate images happen to retain similar colors.
        if any(self.content_changed(p, animations=approved) for p in samples[-8:]):
            return ()
        return tuple(approved)

    def content_changed(self, pixels, *, targets=(), cursor=None, animations=()):
        """Compare CLEAN underlying source pixels, never an overlay screenshot.

        Input activity alone is not a change. Small hover fills, a blinking
        caret, status text and incidental tooltips can repaint without moving
        the objects being explained. Substantial source repaint or displaced
        edges retires the plan; marked targets get a tighter structural check.
        Callers debounce pixel changes and check source HWND/bounds separately.
        This gate has no authority to click or to infer an action succeeded.
        """
        baseline = self.content_fingerprint
        if baseline is None:
            # Older manually constructed frames retain their smaller signature.
            baseline = self.fingerprint
        try:
            current = np.asarray(Image.fromarray(pixels).convert("L").resize(
                (baseline.shape[1], baseline.shape[0])), dtype=np.int16)
        except (AttributeError, TypeError, ValueError):
            return True
        if baseline.shape != current.shape:
            return True
        rows, columns = current.shape
        mon = self.monitor

        def region(rect):
            return _region(rect, mon, (rows, columns))

        source = region(self.window or (mon["left"], mon["top"], mon["width"], mon["height"]))
        if source is None:
            return True
        mask = np.zeros(current.shape, dtype=bool)
        mask[source] = True
        count = int(mask.sum())
        raw = np.abs(current - baseline) > 18
        old_edges, new_edges = _edges(baseline), _edges(current)
        moved_edges = (old_edges & ~_nearby(new_edges)) | (new_edges & ~_nearby(old_edges))

        # Only temporally observed, measured image canvases receive a motion
        # allowance. Their palette and boundary must still match, and a mark
        # must enclose the full canvas; a moving triangle edge is not stable.
        motion = np.zeros(current.shape, dtype=bool)
        containers = []
        for rect in list(animations)[:16]:
            animated = region(rect)
            if animated is None or baseline[animated].size > count * .20:
                return True
            if not _animation_stable(baseline, current, animated):
                return True
            # The boundary/palette check above observes the full canvas. Do
            # not reapply the static edge rule to moving shapes touching its
            # edge (the Wikipedia rearrangement animation does exactly that).
            motion[animated] = True
            containers.append(animated)
        if motion.sum() > count * .25:
            return True
        raw &= ~motion
        moved_edges &= ~motion

        # Cursor-local cosmetic repaint is bounded; structural evidence and
        # every target remain visible to their separate checks below.
        raw_mask = mask.copy()
        if cursor is not None:
            try:
                x, y = map(number, cursor)
                hover = region((x - 48, y - 32, 96, 64))
            except (TypeError, ValueError):
                hover = None
            if hover is not None and not moved_edges[hover].any():
                raw_mask[hover] = False

        if (raw & raw_mask).sum() > max(32, count * .05):
            return True
        # Small new text (browser status/tooltip) is incidental away from a
        # grounded target. A source layout change produces many displaced edges.
        if (moved_edges & mask).sum() > max(900, count * .008):
            return True
        if _layout_replaced(moved_edges & mask):
            return True

        # No mark bounding box is hidden. A fill-only hover on a button keeps
        # its edge/text layout; a moved control or changed value does not.
        reversed_edges = _reversed_edges(baseline, current)
        for rect in targets:
            target = region(rect)
            if target is None:
                continue
            target_mask = mask[target]
            ty, tx = target
            for ay, ax in containers:
                overlap = (tx.start < ax.stop and tx.stop > ax.start
                           and ty.start < ay.stop and ty.stop > ay.start)
                encloses = (tx.start <= ax.start and tx.stop >= ax.stop
                            and ty.start <= ay.start and ty.stop >= ay.stop)
                if overlap and not encloses:
                    return True
            target_raw = raw[target] & target_mask
            area = int(target_mask.sum())
            if not area or target_raw.sum() < 4:
                continue
            old, new = old_edges[target], new_edges[target]
            observable = target_mask & ~motion[target]
            edge_change = (old ^ new) & observable
            edge_count = int((old & target_mask).sum())
            if edge_change.sum() >= max(6, min(32, edge_count * .12)):
                return True
            if (reversed_edges[target] & observable).sum() >= max(4, min(16, edge_count * .03)):
                return True
            # A nearly flat target can change state without acquiring an edge.
            # Small area changes are tolerated globally, but not a substantial
            # replacement of a specifically grounded object.
            if edge_count < 4 and target_raw.sum() > max(16, area * .5):
                return True
        return False

    def unchanged(self, pixels, scene, occlusions=(), scale=1):
        """Legacy overlay-bearing comparison retained for older checks/callers.

        Masked areas cannot prove content identity. Current presentations use
        content_changed on clean underlying pixels, including marked regions;
        ordinary cursor or input activity does not invalidate those drawings.
        """
        current = thumbnail(pixels)
        mon = self.monitor
        mask = np.zeros(current.shape, dtype=bool)
        x,y,w,h = self.window or (mon["left"],mon["top"],mon["width"],mon["height"])
        l=max(0,min(240,math.ceil((x-mon["left"])/mon["width"]*240)))
        t=max(0,min(135,math.ceil((y-mon["top"])/mon["height"]*135)))
        r=max(0,min(240,int((x+w-mon["left"])/mon["width"]*240)))
        b=max(0,min(135,int((y+h-mon["top"])/mon["height"]*135)))
        mask[t:b,l:r]=True
        source_pixels = int(mask.sum())
        for x, y, w, h in occlusions:
            # Thumbnail resampling spreads the edge of our own UI a little.
            px, py = mon["width"] / 120, mon["height"] / 67.5
            x, y, w, h = x-px, y-py, w+2*px, h+2*py
            l = max(0, min(240, int((x - mon["left"]) / mon["width"] * 240)))
            t = max(0, min(135, int((y - mon["top"]) / mon["height"] * 135)))
            r = max(0, min(240, math.ceil((x + w - mon["left"]) / mon["width"] * 240)))
            b = max(0, min(135, math.ceil((y + h - mon["top"]) / mon["height"] * 135)))
            mask[t:b, l:r] = False
        for mark in scene["marks"]:
            xs, ys = zip(*mark["points"])
            # Allow halo/arrowhead and label backing, in physical pixels.
            margin = 32 * scale
            if mark["kind"] == "label":
                # Conservatively cover measured-font label clamping at actual
                # DPI, including wide glyphs (the renderer measures the text).
                window = self.window or (mon["left"], mon["top"], mon["width"], mon["height"])
                left = max(mon["left"], window[0]); top = max(mon["top"], window[1])
                edge = min(mon["left"] + mon["width"], window[0] + window[2])
                base = min(mon["top"] + mon["height"], window[1] + window[3])
                width = max(1, min(edge - left - 12 * scale, max(40, len(mark["text"]) * 18 + 20) * scale))
                x = max(left + 6 * scale, min(xs[0], edge - width - 6 * scale))
                y = max(top + 6 * scale, min(ys[0], base - 34 * scale))
                xs, ys = (x, x + width), (y, y + 28 * scale)
            right = max(xs) + margin
            bottom = max(ys) + margin
            l = max(0, int((min(xs) - margin - mon["left"]) / mon["width"] * 240))
            t = max(0, int((min(ys) - margin - mon["top"]) / mon["height"] * 135))
            r = min(240, math.ceil((right - mon["left"]) / mon["width"] * 240))
            b = min(135, math.ceil((bottom - mon["top"]) / mon["height"] * 135))
            mask[t:b, l:r] = False
        # Fail closed when marks obscure too much evidence to compare.
        if not source_pixels or mask.sum() < source_pixels * .5:
            return False
        return bool((np.abs(current - self.fingerprint)[mask] > 18).mean() < .001)
