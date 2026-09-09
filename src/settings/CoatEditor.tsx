/**
 * Click a part of Mellow, pick a colour. The preview is a canvas so the pixel
 * under the cursor names its own part - the atlas is five known colours.
 * Nothing reaches the desktop pet until Settings saves.
 */

import { useEffect, useRef, useState, type MouseEvent } from "react";
import {
  DEFAULT_COAT,
  EDITABLE,
  PRESETS,
  ROLES,
  ROLE_LABEL,
  coatFrom,
  hslToRgb,
  rgbToHex,
  type Coat,
  type Role,
} from "../ui/coat";
import { CELL, drawCell, roleAt } from "../ui/coatApply";
import sheet from "../pet/sprites.json" with { type: "json" };

const IDLE = sheet.frames.idle.cell;
/** 64 art pixels at 5x. Big enough to click an eye without aiming. */
const SCALE = 5;

/** Generated, not a table to maintain: nine hues over four lightnesses, then greys. */
const SWATCHES: string[] = [
  ...[0, 1, 2, 3].flatMap((row) =>
    [0, 1, 2, 3, 4, 5, 6, 7, 8].map((col) =>
      rgbToHex(hslToRgb([col * 40, 62, 80 - row * 17])),
    ),
  ),
  ...[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => rgbToHex(hslToRgb([0, 0, 96 - i * 11]))),
];

const same = (a: Coat, b: Coat) => ROLES.every((role) => a[role] === b[role]);

/** Shading maps to Patch, since it derives from it. The fixed face maps to nothing. */
const SELECTS: Record<Role, Role | null> = {
  cream: "cream",
  tan: "tan",
  brown: "tan",
  dark: null,
  salmon: "salmon",
};


export function CoatEditor({
  coat,
  onChange,
}: {
  coat: Coat;
  onChange: (coat: Coat) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [part, setPart] = useState<Role>("tan");
  const [hover, setHover] = useState<Role | null>(null);

  useEffect(() => {
    const el = canvas.current;
    if (el) drawCell(el, coat, IDLE, SCALE).catch(() => undefined);
  }, [coat]);

  /** Which part the pointer is over, in art pixels rather than screen pixels. */
  const partAt = (event: MouseEvent<HTMLCanvasElement>): Role | null => {
    const box = event.currentTarget.getBoundingClientRect();
    const x = Math.floor(((event.clientX - box.left) / box.width) * CELL);
    const y = Math.floor(((event.clientY - box.top) / box.height) * CELL);
    return roleAt(IDLE, x, y);
  };

  /** Route every edit through coatFrom so shading and ink can never go stale. */
  const paint = (colour: string) => {
    const next = { ...coat, [part]: colour };
    onChange(coatFrom(next.cream, next.tan, next.salmon));
  };

  return (
    <div className="coat">
      <div className="coat-presets" role="group" aria-label="Coat presets">
        {PRESETS.map((preset) => (
          <button
            key={preset.name}
            type="button"
            className={`coat-preset${same(coat, preset.coat) ? " is-on" : ""}`}
            aria-pressed={same(coat, preset.coat)}
            onClick={() => onChange({ ...preset.coat })}
          >
            <span className="coat-preset__chips" aria-hidden="true">
              {ROLES.map((role) => (
                <i key={role} style={{ background: preset.coat[role] }} />
              ))}
            </span>
            {preset.name}
          </button>
        ))}
      </div>

      <div className="coat-editor">
        <div className="coat-stage">
          <canvas
            ref={canvas}
            className="coat-canvas"
            width={CELL * SCALE}
            height={CELL * SCALE}
            aria-hidden="true"
            onClick={(event) => {
              const hit = partAt(event);
              const target = hit && SELECTS[hit];
              if (target) setPart(target);
            }}
            onMouseMove={(event) => setHover(partAt(event))}
            onMouseLeave={() => setHover(null)}
            style={{ cursor: hover && SELECTS[hover] ? "pointer" : "default" }}
          />
          <p className="coat-hint">
            {hover === "dark"
              ? "The eyes and nose stay dark so Mellow's face always reads"
              : hover && SELECTS[hover]
                ? `Click to edit ${ROLE_LABEL[SELECTS[hover]].toLowerCase()}`
                : "Click a part of Mellow"}
          </p>

          {/* A canvas is not keyboard-reachable, so this row is the real selector. */}
          <div className="coat-parts" role="radiogroup" aria-label="Part of Mellow to colour">
            {EDITABLE.map((role) => (
              <label key={role} className={`coat-part${part === role ? " is-on" : ""}`}>
                <input
                  type="radio"
                  name="coat-part"
                  checked={part === role}
                  onChange={() => setPart(role)}
                />
                <i style={{ background: coat[role] }} aria-hidden="true" />
                {ROLE_LABEL[role]}
              </label>
            ))}
          </div>
        </div>

        <div className="coat-picker">
          <h3>
            {ROLE_LABEL[part]} <span>{coat[part]}</span>
          </h3>
          <div className="coat-swatches" role="group" aria-label={`Colour for ${ROLE_LABEL[part]}`}>
            {SWATCHES.map((colour) => (
              <button
                key={colour}
                type="button"
                className={`coat-swatch${coat[part] === colour ? " is-on" : ""}`}
                style={{ background: colour }}
                aria-label={colour}
                aria-pressed={coat[part] === colour}
                onClick={() => paint(colour)}
              />
            ))}
          </div>
          <label className="coat-custom">
            Custom colour
            <input type="color" value={coat[part]} onChange={(e) => paint(e.target.value)} />
          </label>
          <button
            type="button"
            className="button"
            disabled={same(coat, DEFAULT_COAT)}
            onClick={() => onChange({ ...DEFAULT_COAT })}
          >
            Reset to original
          </button>
        </div>
      </div>
    </div>
  );
}
