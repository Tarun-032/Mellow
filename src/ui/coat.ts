/**
 * Mellow's coat colours. Pure maths, no DOM and no image imports, so
 * `node scripts/coat.check.ts` can run it. The DOM half is coatApply.ts.
 *
 * SOURCE mirrors the locked palette in scripts/sprites.py:59-70 and
 * docs/design.md. The generator snaps every atlas pixel to these, which is what
 * makes an exact runtime palette swap possible in the first place.
 */

export type Role = "cream" | "tan" | "brown" | "dark" | "salmon";

export const ROLES: Role[] = ["cream", "tan", "brown", "dark", "salmon"];

/** What each role actually paints, for the settings labels. */
export const ROLE_LABEL: Record<Role, string> = {
  cream: "Fur",
  tan: "Patch",
  brown: "Shading",
  dark: "Eyes & nose",
  salmon: "Blush",
};

/**
 * The parts a person picks. Shading follows the patch and the face is fixed,
 * because those are the two nobody chooses well - a hand-picked shading rarely
 * belongs to its patch, and a tinted face is how Mellow goes invisible.
 */
export const EDITABLE: Role[] = ["cream", "tan", "salmon"];

export type Coat = Record<Role, string>;

export const SOURCE: Coat = {
  cream: "#f5ecdd",
  tan: "#c9824a",
  brown: "#9b532d",
  dark: "#4c2923",
  salmon: "#f18973",
};

/** Writing-pose pencil. Deliberately not a snap target; recolours to itself. */
export const PENCIL = "#e8a83c";

export const DEFAULT_COAT: Coat = { ...SOURCE };

export type Rgb = [number, number, number];

export function hexToRgb(hex: string): Rgb {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function rgbToHex([r, g, b]: Rgb): string {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  return "#" + [r, g, b].map((v) => clamp(v).toString(16).padStart(2, "0")).join("");
}

export function isHex(value: unknown): value is string {
  return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value);
}

type Hsl = [number, number, number]; // h 0-360, s 0-100, l 0-100

export function rgbToHsl([r, g, b]: Rgb): Hsl {
  const [rn, gn, bn] = [r / 255, g / 255, b / 255];
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const l = (max + min) / 2;
  const d = max - min;
  if (d === 0) return [0, 0, l * 100];
  const s = d / (1 - Math.abs(2 * l - 1));
  let h: number;
  if (max === rn) h = ((gn - bn) / d) % 6;
  else if (max === gn) h = (bn - rn) / d + 2;
  else h = (rn - gn) / d + 4;
  return [((h * 60) % 360 + 360) % 360, s * 100, l * 100];
}

export function hslToRgb([h, s, l]: Hsl): Rgb {
  const sn = s / 100;
  const ln = l / 100;
  const c = (1 - Math.abs(2 * ln - 1)) * sn;
  const hp = (((h % 360) + 360) % 360) / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  const [r1, g1, b1] =
    hp < 1 ? [c, x, 0] :
    hp < 2 ? [x, c, 0] :
    hp < 3 ? [0, c, x] :
    hp < 4 ? [0, x, c] :
    hp < 5 ? [x, 0, c] : [c, 0, x];
  const m = ln - c / 2;
  return [(r1 + m) * 255, (g1 + m) * 255, (b1 + m) * 255];
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/**
 * The shading is the step the artist already drew: measured from SOURCE at
 * module load, tan -> brown is dh -5.7, S x1.018, dl -14.7, and applying it back
 * to the shipped tan returns the shipped brown exactly. Measured rather than
 * written down so it stays true if sprites.py's palette ever moves.
 */
const SHADE = (() => {
  const [th, ts, tl] = rgbToHsl(hexToRgb(SOURCE.tan));
  const [bh, bs, bl] = rgbToHsl(hexToRgb(SOURCE.brown));
  return { dh: bh - th, sRatio: bs / ts, dl: bl - tl };
})();

/** The patch's own darker tone. Nobody picks this well by hand, so nobody does. */
export function shadeOf(patch: string): string {
  const [h, s, l] = rgbToHsl(hexToRgb(patch));
  return rgbToHex(
    hslToRgb([h + SHADE.dh, clamp(s * SHADE.sRatio, 0, 100), clamp(l + SHADE.dl, 0, 100)]),
  );
}

/**
 * Eyes, nose and mouth, on every coat but the shipped one. A neutral near-black
 * reads against any patch hue; letting this follow the coat is what made an
 * all-blue Mellow lose its face.
 *
 * DEFAULT_COAT keeps the warmer shipped #4c2923 on purpose. This role also
 * paints the dialogue text and panel chrome (18 places in pet.css), so making it
 * universal would change how released 1.1.0 looks.
 */
export const INK = "#2f2a28";

/** Blush is pink on every cute animal, whatever colour the animal is. */
export const DEFAULT_BLUSH = SOURCE.salmon;

/**
 * The band that keeps Mellow reading as a light dog with coloured patches. The
 * floor is the shipped cream's own lightness, measured rather than picked, so
 * #f5ecdd passes through untouched by construction instead of by coincidence.
 */
export const FUR_L_MIN = rgbToHsl(hexToRgb(SOURCE.cream))[2];
export const FUR_L_MAX = 98;

/**
 * Fur is the largest area on the pet, so its lightness is clamped while hue and
 * saturation pass through untouched. #f5ecdd is L91 and comes back unchanged; a
 * mid-lightness mint becomes a pale mint-white instead of a green body. That
 * difference is the whole reason the first model produced blobs.
 */
export function furFrom(pick: string): string {
  const [h, s, l] = rgbToHsl(hexToRgb(pick));
  return rgbToHex(hslToRgb([h, s, clamp(l, FUR_L_MIN, FUR_L_MAX)]));
}

/** The three picked colours to all five roles. Shading and ink can never go stale. */
export function coatFrom(fur: string, patch: string, blush: string): Coat {
  return {
    cream: furFrom(fur),
    tan: patch,
    brown: shadeOf(patch),
    dark: INK,
    salmon: blush,
  };
}

/** Coats a real dog could have, plus four that are simply cute. */
export const PRESETS: { name: string; coat: Coat }[] = [
  { name: "Original", coat: DEFAULT_COAT },
  ...(
    [
      ["Golden", "#f7f0e2", "#e0a63f"],
      ["Chocolate", "#f2e9df", "#7b4a32"],
      ["Husky", "#f2f4f6", "#6b7f99"],
      ["Rose", "#fcf7f7", "#c4707f"],
      ["Sky", "#eef4fc", "#6f9fd8"],
      ["Lilac", "#f4f1f8", "#8a79c4"],
    ] as const
  ).map(([name, fur, patch]) => ({ name, coat: coatFrom(fur, patch, DEFAULT_BLUSH) })),
];

/** Every colour the atlas can contain: the five roles plus the pencil. */
const SOURCE_RGB: { key: Role | "pencil"; rgb: Rgb }[] = [
  ...ROLES.map((role) => ({ key: role as Role | "pencil", rgb: hexToRgb(SOURCE[role]) })),
  { key: "pencil" as const, rgb: hexToRgb(PENCIL) },
];

/** Which palette entry a pixel belongs to. Drives recolouring and click-to-select. */
export function nearestSource(r: number, g: number, b: number): Role | "pencil" {
  let best = SOURCE_RGB[0];
  let bestDist = Infinity;
  for (const entry of SOURCE_RGB) {
    const dr = r - entry.rgb[0];
    const dg = g - entry.rgb[1];
    const db = b - entry.rgb[2];
    const dist = dr * dr + dg * dg + db * db;
    if (dist < bestDist) {
      bestDist = dist;
      best = entry;
    }
  }
  return best.key;
}

/**
 * Recolour one pixel: nearest palette entry, then keep its offset from that
 * entry. The offset is zero across the snapped atlas, so that path is an exact
 * swap; writing.png blends 75% palette / 25% source, and the offset is what
 * preserves its glasses, pencil and notebook detail.
 */
export function mapPixel(r: number, g: number, b: number, coat: Coat): Rgb {
  const key = nearestSource(r, g, b);
  const from = key === "pencil" ? hexToRgb(PENCIL) : hexToRgb(SOURCE[key]);
  const to = key === "pencil" ? hexToRgb(PENCIL) : hexToRgb(coat[key]);
  return [
    clamp(to[0] + (r - from[0]), 0, 255),
    clamp(to[1] + (g - from[1]), 0, 255),
    clamp(to[2] + (b - from[2]), 0, 255),
  ];
}

/** Reject anything that is not the five roles, each a #rrggbb string. */
export function isCoat(value: unknown): value is Coat {
  if (typeof value !== "object" || value === null) return false;
  const keys = Object.keys(value);
  return keys.length === ROLES.length && ROLES.every((r) => isHex((value as Coat)[r]));
}
