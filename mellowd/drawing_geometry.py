"""Deterministic constructions in physical desktop pixels.

The model may identify a triangle's vertices. These helpers build its related
geometry locally, after those vertices have been converted out of image space.
Nothing here clips, rounds, or independently rescales a constructed shape.
"""
import math


MIN_SIDE_PX = 4.0
MIN_RELATIVE_AREA = 0.01


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("drawing geometry must use finite numbers")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("drawing geometry must use finite numbers") from None
    if not math.isfinite(result):
        raise ValueError("drawing geometry must use finite numbers")
    return result


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("invalid drawing geometry point")
    return (_number(value[0]), _number(value[1]))


def _bounds(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("invalid drawing geometry bounds")
    left, top, width, height = map(_number, value)
    right, bottom = left + width, top + height
    if width <= 0 or height <= 0 or not all(map(math.isfinite, (right, bottom))):
        raise ValueError("invalid drawing geometry bounds")
    return left, top, right, bottom


def _contain(points, bounds):
    if bounds is None:
        return
    left, top, right, bottom = bounds
    if any(not (left <= x <= right and top <= y <= bottom) for x, y in points):
        raise ValueError("constructed drawing geometry leaves its bounds")


def validate_triangle(points, bounds=None):
    """Return three finite, distinct, non-collinear physical vertices.

    Bounds use (left, top, width, height). The dimensionless area guard rejects
    nearly collinear points regardless of the screenshot's resolution or origin.
    """
    if not isinstance(points, (list, tuple)) or len(points) != 3:
        raise ValueError("drawing geometry needs three triangle vertices")
    vertices = tuple(_point(value) for value in points)
    edges = tuple((vertices[(i + 1) % 3][0] - vertices[i][0],
                   vertices[(i + 1) % 3][1] - vertices[i][1]) for i in range(3))
    lengths = tuple(math.hypot(dx, dy) for dx, dy in edges)
    if any(not math.isfinite(length) or length <= MIN_SIDE_PX for length in lengths):
        raise ValueError("drawing geometry triangle sides are too short")
    longest = max(lengths)
    ax, ay = edges[0][0] / longest, edges[0][1] / longest
    bx, by = -edges[2][0] / longest, -edges[2][1] / longest
    if abs(ax * by - ay * bx) < MIN_RELATIVE_AREA:
        raise ValueError("drawing geometry triangle is nearly collinear")
    _contain(vertices, _bounds(bounds))
    return vertices


def square_on_edge(a, b, opposite, bounds=None):
    """Construct the square on AB on the side away from the opposite vertex.

    Its first two vertices are exactly A and B. Swapping A and B changes only the
    ordering of the same physical square, not the side on which it is placed.
    """
    a, b, opposite = validate_triangle((a, b, opposite))
    dx, dy = b[0] - a[0], b[1] - a[1]
    cx, cy = opposite[0] - a[0], opposite[1] - a[1]
    scale = max(math.hypot(dx, dy), math.hypot(cx, cy))
    cross = (dx / scale) * (cy / scale) - (dy / scale) * (cx / scale)
    px, py = (dy, -dx) if cross > 0 else (-dy, dx)
    square = (a, b, (b[0] + px, b[1] + py), (a[0] + px, a[1] + py))
    if not all(math.isfinite(n) for point in square for n in point):
        raise ValueError("drawing geometry must use finite numbers")
    _contain(square, _bounds(bounds))
    return square


def triangle_squares(points, bounds=None):
    """Return outward AB, BC, and CA squares without changing the input triangle.

    The whole construction is rejected when any square does not fit the supplied
    physical bounds. The caller can choose a simpler explanation in that case.
    """
    a, b, c = validate_triangle(points, bounds)
    return (square_on_edge(a, b, c, bounds), square_on_edge(b, c, a, bounds),
            square_on_edge(c, a, b, bounds))
