/**
 * Recolours the generated art at runtime into CSS variables; coat.ts has the
 * maths. Nothing here touches the art on disk - sprites.py stays its only
 * writer (docs/agents.md rule 5).
 */

import { useEffect } from "react";
import { listen } from "@tauri-apps/api/event";

import { API } from "./fields";
import { ROLES, type Coat, type Role, isCoat, mapPixel, nearestSource } from "./coat";
import sheet from "../pet/sprites.json" with { type: "json" };
import spritesUrl from "../pet/sprites.png";
import writingUrl from "../pet/writing.png";
import bubbleUrl from "../pet/bubble.png";
import boneUrl from "../pet/bone.png";

/** CSS variable per role, matching the names pet.css already uses. */
const VAR: Record<Role, string> = {
  cream: "--paper",
  tan: "--tan",
  brown: "--brown",
  dark: "--ink",
  salmon: "--nose",
};

const ART: { variable: string; url: string }[] = [
  { variable: "--coat-sprites", url: spritesUrl },
  { variable: "--coat-writing", url: writingUrl },
  { variable: "--coat-bubble", url: bubbleUrl },
  { variable: "--coat-bone", url: boneUrl },
];

export const CELL: number = sheet.cell;

function load(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`could not load ${url}`));
    img.src = url;
  });
}

/** Source pixels, kept unrecoloured so hit-testing always asks the same question. */
const pixels = new Map<string, Promise<ImageData>>();
/** The same data once it has resolved, because roleAt has to answer during a click. */
const ready = new Map<string, ImageData>();

function source(url: string): Promise<ImageData> {
  let cached = pixels.get(url);
  if (!cached) {
    cached = load(url).then((img) => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, img.width, img.height);
      ready.set(url, data);
      return data;
    });
    pixels.set(url, cached);
  }
  return cached;
}

/** A recoloured copy. Transparent pixels stay transparent; alpha is untouched. */
function repaint(from: ImageData, coat: Coat): ImageData {
  const out = new ImageData(new Uint8ClampedArray(from.data), from.width, from.height);
  const d = out.data;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] === 0) continue;
    const [r, g, b] = mapPixel(d[i], d[i + 1], d[i + 2], coat);
    d[i] = r;
    d[i + 1] = g;
    d[i + 2] = b;
  }
  return out;
}

async function recolour(url: string, coat: Coat): Promise<string> {
  const data = repaint(await source(url), coat);
  const canvas = document.createElement("canvas");
  canvas.width = data.width;
  canvas.height = data.height;
  canvas.getContext("2d")!.putImageData(data, 0, 0);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) throw new Error(`could not encode ${url}`);
  return URL.createObjectURL(blob);
}

/** Blob URLs currently in use, per element, so superseded ones get revoked. */
const live = new WeakMap<HTMLElement, string[]>();
/** Which applyCoat call owns an element: two can be in flight (StrictMode runs
 *  effects twice), and the slower one would revoke the URLs in use. */
const turn = new WeakMap<HTMLElement, number>();

/**
 * Solid colours land immediately, recoloured art a frame or two later. Pass
 * document.documentElement for the whole window, or a wrapper to scope it.
 */
export async function applyCoat(coat: Coat, el: HTMLElement): Promise<void> {
  for (const role of ROLES) el.style.setProperty(VAR[role], coat[role]);
  const mine = (turn.get(el) ?? 0) + 1;
  turn.set(el, mine);
  const urls = await Promise.all(ART.map((art) => recolour(art.url, coat)));
  if (turn.get(el) !== mine) {
    // A newer coat already landed. Ours is stale, so throw it away rather than
    // painting over the newer one.
    for (const url of urls) URL.revokeObjectURL(url);
    return;
  }
  ART.forEach((art, i) => el.style.setProperty(art.variable, `url(${urls[i]})`));
  for (const old of live.get(el) ?? []) URL.revokeObjectURL(old);
  live.set(el, urls);
}

/** Draw one recoloured atlas cell, scaled up with the pixel grid intact. */
export async function drawCell(
  canvas: HTMLCanvasElement,
  coat: Coat,
  cell: number,
  scale: number,
): Promise<void> {
  const data = repaint(await source(spritesUrl), coat);
  const frame = document.createElement("canvas");
  frame.width = data.width;
  frame.height = data.height;
  frame.getContext("2d")!.putImageData(data, 0, 0);
  canvas.width = CELL * scale;
  canvas.height = CELL * scale;
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(frame, cell * CELL, 0, CELL, CELL, 0, 0, canvas.width, canvas.height);
}

/**
 * The part at an art pixel. Asks the source pixels so the answer does not drift
 * with the coat. null means transparent, pencil, or not loaded yet.
 */
export function roleAt(cell: number, x: number, y: number): Role | null {
  const data = ready.get(spritesUrl);
  if (!data || x < 0 || y < 0 || x >= CELL || y >= CELL) return null;
  const i = (y * data.width + (cell * CELL + x)) * 4;
  if (data.data[i + 3] === 0) return null;
  const role = nearestSource(data.data[i], data.data[i + 1], data.data[i + 2]);
  return role === "pencil" ? null : role;
}

/** Read the saved coat on mount and follow it when Settings saves a new one. */
export function useCoat(): void {
  useEffect(() => {
    let alive = true;
    const paint = (coat: unknown) => {
      if (alive && isCoat(coat)) applyCoat(coat, document.documentElement).catch(() => {});
    };
    fetch(`${API}/config`)
      .then((r) => r.json())
      .then((body) => paint(body?.settings?.coat))
      .catch(() => {});
    // guide-bubble is not in src-tauri/capabilities, so listen can reject there.
    // It still gets the saved coat from the fetch above.
    const stop = listen<Coat>("coat", (event) => paint(event.payload)).catch(
      () => () => undefined,
    );
    return () => {
      alive = false;
      stop.then((off) => off()).catch(() => {});
    };
  }, []);
}
