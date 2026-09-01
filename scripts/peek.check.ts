/** Offline peek geometry check: `node scripts/peek.check.ts` */
import assert from "node:assert/strict";
import sprites from "../src/pet/sprites.json" with { type: "json" };
import { edgeAt, peekBox, peekX, type Side } from "../src/pet/usePetMotion.ts";

const SIDES: Side[] = ["left", "right"];
const WIDTHS = [1280, 1920, 2560, 3840];
const SCALE = 3;
const CELL = sprites.cell * SCALE;
const ART_X = sprites.body.x * SCALE;
const ART_W = sprites.body.w * SCALE;

// Peek art must be flush with the right of its cell (`edge=True` in sprites.py).
assert.equal(
  sprites.peek.x + sprites.peek.w,
  sprites.cell,
  "the peek art must be flush with the right of its cell",
);
// Peek crop must stay a sliver of the cell.
assert.ok(
  sprites.peek.w < sprites.cell * 0.6,
  `peek art is ${sprites.peek.w} of ${sprites.cell} wide — too much dog`,
);

for (const width of WIDTHS) {
  // Standing dog fully on screen must not trigger tuck.
  assert.equal(edgeAt(width / 2, width), null, `middle of ${width}`);
  assert.equal(edgeAt(0, width), null, "flush left is still fully visible");
  assert.equal(edgeAt(width - CELL, width), null, "flush right likewise");

  // Trigger on art crossing the edge, not the cell bounds.
  assert.equal(edgeAt(-ART_X, width), null, "art exactly touching the edge");
  assert.equal(edgeAt(-ART_X - 1, width), "left", "one pixel past it");
  assert.equal(edgeAt(width - ART_X - ART_W, width), null, "right, touching");
  assert.equal(edgeAt(width - ART_X - ART_W + 1, width), "right", "one past");

  // Tucked: whole cell stays on screen.
  assert.equal(peekX("left", width), 0);
  assert.equal(peekX("right", width), width - CELL);

  const widths = SIDES.map((side) => {
    const box = peekBox(side, peekX(side, width), 0);
    // Drawn edge flush with the screen edge.
    assert.equal(
      side === "left" ? box.x0 : box.x1,
      side === "left" ? 0 : width,
      `${side} of ${width} is not flush with the edge`,
    );
    // Entire drawn sliver stays on screen.
    assert.ok(box.x0 >= 0 && box.x1 <= width, `${side} of ${width} runs off`);
    assert.ok(box.y1 > box.y0, "the box has height");
    return box.x1 - box.x0;
  });
  assert.equal(widths[0], widths[1], `${width} shows the same dog either side`);
  assert.ok(widths[0] > 60, `only ${widths[0]}px of dog shows — too little`);
}

console.log("ok  peek art is flush with the edge, fully on screen, same both sides");
