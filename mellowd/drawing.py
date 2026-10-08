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
        if any(self.content_changed(p, targets=approved) for p in samples[-8:]):
            return ()
        return tuple(approved)

    def live_regions(self, samples, *, unattended=False):
        """Physical rects of areas that changed by themselves (terminal output,
        a video). `unattended`: one change is enough. See docs/decisions.md."""
        baseline = self.content_fingerprint
        if baseline is None:
            return ()
        frames = [baseline, *(np.asarray(s, dtype=np.int16) for s in samples if np.shape(s) == baseline.shape)]
        if len(frames) < 3:
            return ()
        rows, columns = baseline.shape
        pad = ((0, (-rows) % 8), (0, (-columns) % 8))

        def tiles(before, after):
            changed = np.pad(np.abs(after - before) > 18, pad)
            return changed.reshape(changed.shape[0] // 8, 8, changed.shape[1] // 8, 8).sum(axis=(1, 3)) >= 2

        counts = sum(tiles(a, b).astype(np.int8) for a, b in zip(frames, frames[1:]))
        seeds, live = counts >= (1 if unattended else 2), counts >= 1
        # Exact changed pixels, so an area ends where its content does.
        changed_pixels = np.any([np.abs(b - a) > 18 for a, b in zip(frames, frames[1:])], axis=0)
        mon = self.monitor
        source = _region(self.window or (mon["left"], mon["top"], mon["width"], mon["height"]), mon, (rows, columns))
        if source is None:
            return ()
        sy, sx = source
        inside = np.zeros_like(live)
        inside[sy.start // 8:-(-sy.stop // 8), sx.start // 8:-(-sx.stop // 8)] = True
        live &= inside
        seeds &= inside
        scale_x, scale_y = mon["width"] / columns, mon["height"] / rows
        regions, seen = [], np.zeros_like(live)
        for start in zip(*np.nonzero(seeds)):
            if seen[start]:
                continue
            seen[start] = True
            pending, cells = [start], []
            while pending:
                y, x = pending.pop()
                cells.append((y, x))
                # Touching tiles only, so a tab title beside the pane stays separate.
                for ny in range(max(0, y - 1), min(live.shape[0], y + 2)):
                    for nx in range(max(0, x - 1), min(live.shape[1], x + 2)):
                        if live[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            pending.append((ny, nx))
            if len(cells) < 6:
                continue   # a caret or spinner
            area = np.zeros(changed_pixels.shape, dtype=bool)
            for y, x in cells:
                area[y * 8:y * 8 + 8, x * 8:x * 8 + 8] = True
            ys, xs = np.nonzero(area & changed_pixels)
            # Margin: an edge flush with changing content would keep changing.
            top, bottom = max(sy.start, int(ys.min()) - 2), min(sy.stop, int(ys.max()) + 3)
            left, right = max(sx.start, int(xs.min()) - 2), min(sx.stop, int(xs.max()) + 3)
            box = (round(mon["left"] + left * scale_x), round(mon["top"] + top * scale_y),
                   round((right - left) * scale_x), round((bottom - top) * scale_y))
            if not any(x <= box[0] and y <= box[1] and box[0] + box[2] <= x + w and box[1] + box[3] <= y + h
                       for x, y, w, h in regions):
                regions.append(box)
        return tuple(regions)

    def content_changed(self, pixels, *, targets=()):
        """Is every marked target still where it was (clean pixels, per target)?
        Returns "moved", "gone", "unverifiable" or False. See docs/decisions.md."""
        baseline = self.content_fingerprint
        if baseline is None:
            # Older manually constructed frames retain their smaller signature.
            baseline = self.fingerprint
        try:
            current = np.asarray(Image.fromarray(pixels).convert("L").resize(
                (baseline.shape[1], baseline.shape[0])), dtype=np.int16)
        except (AttributeError, TypeError, ValueError):
            return "unreadable"
        if baseline.shape != current.shape:
            return "unreadable"
        rows, columns = current.shape
        mon = self.monitor
        source = _region(self.window or (mon["left"], mon["top"], mon["width"], mon["height"]), mon, (rows, columns))
        if source is None:
            return "source"
        if not targets:
            return False
        window = np.zeros(current.shape, dtype=bool)
        window[source] = True
        raw = np.abs(current - baseline) > 18
        old_edges, new_edges = _edges(baseline), _edges(current)
        reversed_edges = _reversed_edges(baseline, current)
        # Structure gone from where it was; new lines appearing are not movement.
        lost = old_edges & ~_nearby(new_edges)

        def differs(area):
            """Did the content change, beyond a hover fill?"""
            changed = raw & area
            if changed.sum() < 4:
                return False
            edge_count = int((old_edges & area).sum())
            if ((old_edges ^ new_edges) & area).sum() >= max(6, min(32, edge_count * .12)):
                return True
            if (reversed_edges & area).sum() >= max(4, min(16, edge_count * .03)):
                return True
            # A nearly flat target can change state without acquiring an edge.
            return edge_count < 4 and changed.sum() > max(16, area.sum() * .5)

        for rect in targets:
            target = _region(rect, mon, (rows, columns))
            if target is None:
                continue
            inner = np.zeros(current.shape, dtype=bool)
            inner[target] = True
            inner &= window
            if not inner.any() or not differs(inner):
                continue
            ty, tx = target
            # Flat where it had structure: the thing vanished.
            core = inner.copy()
            if ty.stop - ty.start > 4 and tx.stop - tx.start > 4:
                core[:ty.start + 2] = core[ty.stop - 2:] = False
                core[:, :tx.start + 2] = core[:, tx.stop - 2:] = False
            had = int((old_edges & core).sum())
            if had >= 6 and (new_edges & core).sum() < had * .1 and current[core].std() < 6:
                return "gone"

            def band(outside, inside):
                ring = np.zeros(current.shape, dtype=bool)
                ring[max(0, ty.start - outside):ty.stop + outside, max(0, tx.start - outside):tx.stop + outside] = True
                ring[ty.start + inside:max(ty.start + inside, ty.stop - inside),
                     tx.start + inside:max(tx.start + inside, tx.stop - inside)] = False
                return ring & window

            # The ring around the target, widened until it has structure.
            for ring in (band(2, 0), band(4, 0), band(8, 0), band(16, 0), band(16, 2)):
                structure = int((old_edges & ring).sum())
                if structure >= 12:
                    break
            else:
                return "unverifiable"
            if ((lost & ring).sum() >= max(6, structure * .12)
                    or (reversed_edges & ring).sum() >= max(4, structure * .03)):
                return "moved"
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
