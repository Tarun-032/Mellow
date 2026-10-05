import type { AnnotationMark, XY } from "./annotationScene";

export type CaptionBox = [number, number, number, number];
export type CaptionBounds = [number, number, number, number];
export type CaptionObstacle = { kind: "box" | "ellipse"; bounds: CaptionBounds }
  | { kind: "stroke"; start: XY; end: XY };

const GAP = 5;

/** Protect drawn edges and selected objects, rather than every polygon's box. */
export function captionObstacles(marks: AnnotationMark[]): CaptionObstacle[] {
  const out: CaptionObstacle[] = [];
  for (const mark of marks) {
    if (mark.kind === "label") continue;
    const points = mark.points;
    if (["rectangle", "highlight", "ellipse"].includes(mark.kind)) {
      out.push({ kind: mark.kind === "ellipse" ? "ellipse" : "box", bounds: [...points[0], ...points[1]] as CaptionBounds });
      continue;
    }
    let path = points;
    if (mark.kind === "quadratic" || mark.kind === "cubic") {
      // Sampling is for caption clearance only; SVG remains the exact curve.
      path = Array.from({ length: 49 }, (_, index): XY => {
        const t = index / 48, u = 1 - t;
        return [0, 1].map((axis) => mark.kind === "quadratic"
          ? u * u * points[0][axis] + 2 * u * t * points[1][axis] + t * t * points[2][axis]
          : u * u * u * points[0][axis] + 3 * u * u * t * points[1][axis]
            + 3 * u * t * t * points[2][axis] + t * t * t * points[3][axis]) as XY;
      });
    }
    for (let index = 1; index < path.length; index++) out.push({ kind: "stroke", start: path[index - 1], end: path[index] });
    if (mark.kind === "polygon") out.push({ kind: "stroke", start: points[points.length - 1], end: points[0] });
    if (mark.kind === "arrow") {
      const [x, y] = points[points.length - 1];
      out.push({ kind: "box", bounds: [x - 11, y - 11, x + 11, y + 11] });
    }
  }
  return out;
}

function strokeIntersects(start: XY, end: XY, [left, top, right, bottom]: CaptionBounds): boolean {
  const dx = end[0] - start[0], dy = end[1] - start[1];
  let near = 0, far = 1;
  for (const [p, q] of [[-dx, start[0] - left], [dx, right - start[0]], [-dy, start[1] - top], [dy, bottom - start[1]]]) {
    if (p === 0) { if (q < 0) return false; continue; }
    const t = q / p;
    if (p < 0) near = Math.max(near, t);
    else far = Math.min(far, t);
    if (near > far) return false;
  }
  return true;
}

export function captionIntersects([x, y, width, height]: CaptionBox, obstacle: CaptionObstacle): boolean {
  const area: CaptionBounds = [x - GAP, y - GAP, x + width + GAP, y + height + GAP];
  if (obstacle.kind === "stroke") return strokeIntersects(obstacle.start, obstacle.end, area);
  const [left, top, right, bottom] = obstacle.bounds;
  if (area[0] >= right || area[2] <= left || area[1] >= bottom || area[3] <= top) return false;
  if (obstacle.kind === "box") return true;
  const rx = (right - left) / 2, ry = (bottom - top) / 2, cx = left + rx, cy = top + ry;
  if (rx <= 0 || ry <= 0) return false;
  const closestX = Math.max(area[0], Math.min(cx, area[2]));
  const closestY = Math.max(area[1], Math.min(cy, area[3]));
  return ((closestX - cx) / rx) ** 2 + ((closestY - cy) / ry) ** 2 <= 1;
}

/** A caption stays beside the bone, inside the source, and off useful ink. */
export function placeCaption(tip: XY, width: number, bounds: CaptionBounds, obstacles: CaptionObstacle[]): CaptionBox | null {
  const [left, top, right, bottom] = bounds, height = 28, margin = 6;
  if (!Number.isFinite(width) || width < 48 || right - left < width + margin * 2 || bottom - top < height + margin * 2) return null;
  const positions: XY[] = [
    [tip[0] + 22, tip[1] - 34], [tip[0] - width - 18, tip[1] - 34],
    [tip[0] + 22, tip[1] + 14], [tip[0] - width - 18, tip[1] + 14],
    [tip[0] - width / 2, tip[1] - height - 18], [tip[0] - width / 2, tip[1] + 18],
    [tip[0] + 22, tip[1] - height / 2], [tip[0] - width - 18, tip[1] - height / 2],
  ];
  for (const [x, y] of positions) {
    const box: CaptionBox = [Math.max(left + margin, Math.min(x, right - width - margin)),
      Math.max(top + margin, Math.min(y, bottom - height - margin)), width, height];
    // Clamping at a window edge must not place the caption over the bone.
    if (tip[0] >= box[0] - 12 && tip[0] <= box[0] + width + 12 && tip[1] >= box[1] - 12 && tip[1] <= box[1] + height + 12) continue;
    if (!obstacles.some((obstacle) => captionIntersects(box, obstacle))) return box;
  }
  // Keep the explanation in Mellow's speech/bubble instead of hiding a target.
  return null;
}
