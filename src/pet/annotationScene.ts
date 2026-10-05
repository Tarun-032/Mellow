export type XY = [number, number];
export type AnnotationMark = {
  kind: "rectangle" | "highlight" | "ellipse" | "line" | "arrow" | "polygon" | "label" | "quadratic" | "cubic";
  points: XY[];
  text: string | null;
  color: "mint" | "amber" | "violet";
};
export type AnnotationScene = {
  presentation_id: string; frame_id: string; hwnd: number; window: number[] | null;
  monitor: { left: number; top: number; width: number; height: number };
  lifetime_ms: number; marks: AnnotationMark[]; reveal_from?: number; caption?: string;
};
export type AnnotationDisplay = { revision: number; scene: AnnotationScene; scale: number; pen_origin?: XY };

let revision = 0;
export function nextAnnotationRevision() {
  revision = Math.max(revision + 1, Date.now() * 1000);
  return revision;
}

export function localPoint(point: XY, display: AnnotationDisplay): XY {
  return [(point[0] - display.scene.monitor.left) / display.scale,
    (point[1] - display.scene.monitor.top) / display.scale];
}

export function isAreaMark(kind: AnnotationMark["kind"]) {
  return kind === "rectangle" || kind === "highlight" || kind === "ellipse";
}

/** A selection gesture holds the first corner and drags the opposite corner. */
export function dragPoint(start: XY, end: XY, progress: number): XY {
  const p = Math.max(0, Math.min(1, progress));
  return [start[0] + (end[0] - start[0]) * p, start[1] + (end[1] - start[1]) * p];
}

export function areaPathFor(mark: AnnotationMark, display: AnnotationDisplay, progress: number): string {
  return pathFor({ ...mark, points: [mark.points[0], dragPoint(mark.points[0], mark.points[1], progress)] }, display);
}

/** Cache the complete trace, independently of the SVG path being revealed. */
export function tracePoints(mark: AnnotationMark, display: AnnotationDisplay): XY[] | null {
  if (mark.kind !== "line" && mark.kind !== "polygon") return null;
  const points = mark.points.map((point) => localPoint(point, display));
  return mark.kind === "polygon" ? [...points, points[0]] : points;
}

export function traceLength(points: XY[]): number {
  return points.slice(1).reduce((total, end, index) => total + Math.hypot(end[0] - points[index][0], end[1] - points[index][1]), 0);
}

/** Reveal an open prefix; the visible stroke and bone end at exactly one tip.
 * A dashed closed SVG can wrap a second dash around its closing corner.
 */
export function traceAt(points: XY[], progress: number): { path: string; tip: XY } {
  if (progress >= 1) return { path: `M${points.map(([x, y]) => `${x},${y}`).join(" L")}`, tip: points[points.length - 1] };
  const lengthTotal = traceLength(points), epsilon = Math.max(1, lengthTotal) * 1e-12;
  let remaining = lengthTotal * Math.max(0, Math.min(1, progress));
  const visible = [points[0]];
  for (let index = 1; index < points.length; index++) {
    if (remaining <= epsilon) break;
    const start = points[index - 1], end = points[index];
    const length = Math.hypot(end[0] - start[0], end[1] - start[1]);
    if (remaining + epsilon >= length) {
      visible.push(end); remaining -= length;
    } else {
      if (remaining > 0) visible.push(dragPoint(start, end, remaining / length));
      break;
    }
  }
  return { path: `M${visible.map(([x, y]) => `${x},${y}`).join(" L")}`, tip: visible[visible.length - 1] };
}

/** Geometry is built from typed points, never arbitrary SVG/path input. */
export function pathFor(mark: AnnotationMark, display: AnnotationDisplay): string {
  const p = mark.points.map((point) => localPoint(point, display));
  const pair = ([x, y]: XY) => `${x},${y}`;
  if (mark.kind === "rectangle" || mark.kind === "highlight") {
    const [[x, y], [right, bottom]] = p;
    return `M${x},${y} H${right} V${bottom} H${x} Z`;
  }
  if (mark.kind === "ellipse") {
    const [[x, y], [right, bottom]] = p;
    const rx = (right - x) / 2, ry = (bottom - y) / 2, cy = y + ry;
    if (rx <= 0 || ry <= 0) return `M${x},${y}`;
    return `M${x},${cy} a${rx},${ry} 0 1,0 ${rx * 2},0 a${rx},${ry} 0 1,0 ${-rx * 2},0`;
  }
  if (mark.kind === "quadratic") return `M${pair(p[0])} Q${pair(p[1])} ${pair(p[2])}`;
  if (mark.kind === "cubic") return `M${pair(p[0])} C${pair(p[1])} ${pair(p[2])} ${pair(p[3])}`;
  return `M${p.map(pair).join(" L")}${mark.kind === "polygon" ? " Z" : ""}`;
}

export function arrowhead(mark: AnnotationMark, display: AnnotationDisplay, progress = 1): string {
  const [a, tip] = mark.points.map((point) => localPoint(point, display));
  const b = dragPoint(a, tip, progress);
  const angle = Math.atan2(tip[1] - a[1], tip[0] - a[0]);
  // Keep the head at the dragging bone. It grows with the shaft and never
  // becomes a second path for the bone to walk back along.
  const size = Math.min(11, Math.hypot(b[0] - a[0], b[1] - a[1]) * .35);
  const end = (offset: number) => [b[0] - size * Math.cos(angle + offset), b[1] - size * Math.sin(angle + offset)].join(",");
  return `M${end(.45)} L${b.join(",")} L${end(-.45)}`;
}

export function arrowPathFor(mark: AnnotationMark, display: AnnotationDisplay, progress: number): string {
  return pathFor({ ...mark, points: [mark.points[0], dragPoint(mark.points[0], mark.points[1], progress)] }, display);
}
