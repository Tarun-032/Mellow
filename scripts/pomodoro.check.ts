/** Offline pomodoro arithmetic check: `node scripts/pomodoro.check.ts` */
import assert from "node:assert/strict";
import { LIMITS, advance, clampSettings, mmss, type Phase } from "../src/pet/usePomodoro.ts";

/** Walk a session via advance(); return the path. */
function walk(rounds: number, cap = 40) {
  const path: string[] = [];
  let at: { phase: Phase; round: number } | null = { phase: "focus", round: 1 };
  while (at !== null) {
    path.push(`${at.phase}/${at.round}`);
    assert.ok(path.length <= cap, `session never ended: ${path.join(" ")}`);
    at = advance(at.phase, at.round, rounds);
  }
  return path;
}

// One round: focus only, no trailing break.
assert.deepEqual(walk(1), ["focus/1"]);

// Multi-round sessions must end after the last focus.
assert.deepEqual(walk(3), [
  "focus/1",
  "break/1",
  "focus/2",
  "break/2",
  "focus/3",
]);

// Break keeps round; next focus increments.
assert.deepEqual(advance("focus", 1, 4), { phase: "break", round: 1 });
assert.deepEqual(advance("break", 1, 4), { phase: "focus", round: 2 });
assert.equal(advance("focus", 4, 4), null);
// Past the end still ends.
assert.equal(advance("focus", 9, 4), null);

// Junk settings fall back to defaults.
const fallback = { focus: 25, break: 5, rounds: 4 };
assert.deepEqual(clampSettings(null), fallback);
assert.deepEqual(clampSettings({}), fallback);
assert.deepEqual(clampSettings("nonsense"), fallback);
assert.deepEqual(clampSettings({ focus: 0 }), fallback);
assert.deepEqual(clampSettings({ focus: 900 }), fallback);
assert.deepEqual(clampSettings({ rounds: "4" }), fallback);
assert.deepEqual(clampSettings({ focus: NaN }), fallback);
// Bad keys dropped; good siblings kept.
assert.deepEqual(clampSettings({ focus: 50, rounds: 99 }), {
  ...fallback,
  focus: 50,
});

// Every stepper value must round-trip.
for (const key of Object.keys(LIMITS) as (keyof typeof LIMITS)[]) {
  const [low, high, step] = LIMITS[key];
  for (let value = low; value <= high; value += step) {
    assert.equal(clampSettings({ [key]: value })[key], value);
  }
  assert.equal(clampSettings({ [key]: low - step })[key], fallback[key]);
  assert.equal(clampSettings({ [key]: high + step })[key], fallback[key]);
}

// mm:ss stays fixed width.
assert.equal(mmss(0), "00:00");
assert.equal(mmss(59), "00:59");
assert.equal(mmss(60), "01:00");
assert.equal(mmss(25 * 60), "25:00");
assert.equal(mmss(60 * 60), "60:00");
// Negative remaining clamps to 00:00.
assert.equal(mmss(-5), "00:00");

console.log("ok  pomodoro sessions end, settings survive junk, clock stays 5 wide");
