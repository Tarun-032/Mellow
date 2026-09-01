/** Offline bone geometry check: `node scripts/bone.check.ts` */
import assert from "node:assert/strict";
import sprites from "../src/pet/sprites.json" with { type: "json" };
import { bonePlacement } from "../src/pet/usePetMotion.ts";

const SCALE = 3;
const SCREENS = [
  [1280, 720],
  [1920, 1080],
  [2560, 1440],
  [3840, 2160],
];

// Hotspot must lie inside the bone art.
assert.ok(
  sprites.bone.tip.x < sprites.bone.w && sprites.bone.tip.y < sprites.bone.h,
  `tip ${JSON.stringify(sprites.bone.tip)} is outside a ${sprites.bone.w}x${sprites.bone.h} bone`,
);
// Hotspot in the leading quarter (not the belly).
assert.ok(
  sprites.bone.tip.x < sprites.bone.w / 2 &&
    sprites.bone.tip.y < sprites.bone.h / 2,
  "the hotspot must be in the leading quarter of the bone",
);
// Rendered size should read as a pointer (~30–42px at 3x).
const rendered = sprites.bone.w * SCALE;
assert.ok(
  rendered >= 30 && rendered <= 42,
  `${rendered}px is not a pointer-sized bone`,
);

for (const [w, h] of SCREENS) {
  // Tip must land exactly on the fractional target.
  for (const [nx, ny] of [
    [0, 0],
    [0.5, 0.5],
    [1, 1],
    [0.25, 0.8],
  ]) {
    const spot = bonePlacement({ nx, ny }, w, h);
    assert.equal(
      spot.x + sprites.bone.tip.x * SCALE,
      nx * w,
      `tip missed x on ${w}x${h}`,
    );
    assert.equal(
      spot.y + sprites.bone.tip.y * SCALE,
      ny * h,
      `tip missed y on ${w}x${h}`,
    );
  }

  // Out-of-range targets clamp to the screen edge.
  const off = bonePlacement({ nx: 2.5, ny: -1 }, w, h);
  assert.equal(off.x + sprites.bone.tip.x * SCALE, w, "an over-range x escaped");
  assert.equal(off.y + sprites.bone.tip.y * SCALE, 0, "an under-range y escaped");

  // Bubble opens away from the nearer edge.
  assert.equal(bonePlacement({ nx: 0.1, ny: 0.1 }, w, h).side, "right");
  assert.equal(bonePlacement({ nx: 0.9, ny: 0.1 }, w, h).side, "left");
  assert.equal(bonePlacement({ nx: 0.1, ny: 0.1 }, w, h).lift, "below");
  assert.equal(bonePlacement({ nx: 0.1, ny: 0.9 }, w, h).lift, "above");
}

console.log(
  `ok  bone tip lands on target across ${SCREENS.length} screens, clamps, and flips its bubble at the edges`,
);
