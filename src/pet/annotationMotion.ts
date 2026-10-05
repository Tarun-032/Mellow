import type { XY } from "./annotationScene";

export type Stroke = { index: number; start: XY; end: XY; length: number; caption: string; gesture?: "area" | "arrow" };
export type StrokeStep = Stroke & { from: XY; begin: number; travel: number; draw: number; hold: number };
/** Leave time within the native 60-second lease for placement and receipts. */
export const MAX_STROKE_TIMELINE_MS = 55_000;
export const DRAW_PACE = 1.35;
export function strokeDuration(steps: StrokeStep[]): number {
  const last = steps[steps.length - 1];
  return last ? last.begin + last.travel + last.draw + last.hold : 0;
}
export function strokeRemaining(steps: StrokeStep[], elapsed: number): number {
  return Math.min(MAX_STROKE_TIMELINE_MS, Math.ceil(Math.max(0, strokeDuration(steps) - elapsed)));
}
export function strokeSequence(strokes: Stroke[], origin: XY): StrokeStep[] {
  let at = origin, clock = 0;
  const steps = strokes.map((stroke, index) => {
    const distance = Math.hypot(stroke.start[0] - at[0], stroke.start[1] - at[1]);
    const travel = distance < 1 ? 0 : Math.min(index === 0 ? 850 : 500, Math.max(180, distance / 1.6));
    const draw = DRAW_PACE * (stroke.length === 0 ? 0 : stroke.gesture === "arrow"
      ? Math.min(320, Math.max(180, stroke.length / 1.8)) : stroke.gesture === "area"
      ? Math.min(480, Math.max(260, stroke.length / 2))
      : Math.min(1600, Math.max(650, stroke.length / .7)));
    const hold = stroke.gesture && stroke.caption && index < strokes.length - 1
      ? Math.min(700, Math.max(350, stroke.caption.length * 22)) : 0;
    const step = { ...stroke, from: at, begin: clock, travel, draw, hold };
    clock += travel + draw + hold;
    at = stroke.end;
    return step;
  });
  if (clock > MAX_STROKE_TIMELINE_MS) {
    const speed = MAX_STROKE_TIMELINE_MS / clock;
    for (const step of steps) {
      step.begin *= speed; step.travel *= speed; step.draw *= speed; step.hold *= speed;
    }
  }
  return steps;
}
export function strokeClock(steps: StrokeStep[], elapsed: number) {
  for (let index = 0; index < steps.length; index++) {
    const step = steps[index];
    if (elapsed < step.begin + step.travel) {
      const p = Math.max(0, (elapsed - step.begin) / step.travel);
      const smooth = p * p * p * (p * (p * 6 - 15) + 10);
      return { index, phase: "travel" as const, progress: 0,
        tip: [step.from[0] + (step.start[0] - step.from[0]) * smooth,
          step.from[1] + (step.start[1] - step.from[1]) * smooth] as XY };
    }
    if (elapsed < step.begin + step.travel + step.draw) {
      return { index, phase: "draw" as const, progress: (elapsed - step.begin - step.travel) / step.draw, tip: step.start };
    }
    if (elapsed < step.begin + step.travel + step.draw + step.hold) {
      return { index, phase: "hold" as const, progress: 1, tip: step.end };
    }
  }
  return { index: Math.max(0, steps.length - 1), phase: "done" as const, progress: 1,
    tip: steps[steps.length - 1]?.end ?? [0, 0] as XY };
}
