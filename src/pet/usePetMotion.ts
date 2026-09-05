import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
// Import attribute so peek.check.ts can load this under plain Node.
import sprites from "./sprites.json" with { type: "json" };
import { spring, stepSpring } from "./spring.ts";

const SCALE = 3;
const CELL = sprites.cell * SCALE;
const BODY = {
  x: sprites.body.x * SCALE,
  y: sprites.body.y * SCALE,
  w: sprites.body.w * SCALE,
  h: sprites.body.h * SCALE,
};
const MARGIN_X = 24;
const MARGIN_Y = 80;
/** Peek pose box inside the cell (not inside BODY). */
const PEEK = {
  x: sprites.peek.x * SCALE,
  y: sprites.peek.y * SCALE,
  w: sprites.peek.w * SCALE,
  h: sprites.peek.h * SCALE,
};
/** Bone size and tip hotspot from sprites.json. */
const BONE = {
  w: sprites.bone.w * SCALE,
  h: sprites.bone.h * SCALE,
  tipX: sprites.bone.tip.x * SCALE,
  tipY: sprites.bone.tip.y * SCALE,
};
/** Bone target as a fraction of this (monitor-sized) window. */
export type Target = { nx: number; ny: number };

type Cursor = { x: number; y: number };
export type Reaction = "drag" | "pet" | "hunt" | "angry" | null;
/** Which edge he is tucked behind. Null is "out in the open". */
export type Side = "left" | "right";

type Motion = {
  x: ReturnType<typeof spring>;
  y: ReturnType<typeof spring>;
  sx: ReturnType<typeof spring>;
  sy: ReturnType<typeof spring>;
  angle: ReturnType<typeof spring>;
  cursor: Cursor & { at: number; vx: number; vy: number; speed: number };
  initialized: boolean;
  dragging: boolean;
  dragOffset: Cursor;
  dragAt: number;
  dragCursor: Cursor;
  dragSpeed: number;
  dragVx: number;
  dragSign: number;
  shakeTurns: number[];
  angryUntil: number;
  petUntil: number;
  petEnergy: number;
  petSign: number;
  huntUntil: number;
  huntSign: number;
  huntTurns: number[];
  huntTravel: number;
  huntLastFastAt: number;
  landUntil: number;
  reaction: Reaction;
  quiet: Side | null;
  home: Cursor | null;
};

const clamp = (n: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, n));

/** Bone cell placement and bubble open side (half-screen). */
export function bonePlacement(t: Target, width: number, height: number) {
  return {
    x: clamp(t.nx, 0, 1) * width - BONE.tipX,
    y: clamp(t.ny, 0, 1) * height - BONE.tipY,
    side: t.nx > 0.5 ? "left" : "right",
    lift: t.ny > 0.5 ? "above" : "below",
  } as const;
}

/** Which edge the art was dropped into, if any. */
export function edgeAt(x: number, width: number): Side | null {
  if (x + BODY.x < 0) return "left";
  if (x + BODY.x + BODY.w > width) return "right";
  return null;
}

/** Flush peek position against an edge. */
export function peekX(side: Side, width: number): number {
  return side === "left" ? 0 : width - CELL;
}

/** Peek art box on screen for a cell origin. */
export function peekBox(side: Side, x: number, y: number) {
  const left = side === "left" ? CELL - PEEK.x - PEEK.w : PEEK.x;
  return { x0: x + left, x1: x + left + PEEK.w, y0: y + PEEK.y, y1: y + PEEK.y + PEEK.h };
}

function hitsPet(cursor: Cursor, x: number, y: number, padding: number) {
  const artX = Math.floor((cursor.x - x) / SCALE);
  const artY = Math.floor((cursor.y - y) / SCALE);
  const radius = Math.ceil(padding / SCALE);
  for (let dy = -radius; dy <= radius; dy += 1) {
    const row = sprites.hit[artY + dy];
    if (!row) continue;
    for (let dx = -radius; dx <= radius; dx += 1) {
      if (dx * dx + dy * dy <= radius * radius && row[artX + dx] === "1") {
        return true;
      }
    }
  }
  return false;
}

/** Cursor inside an element's client box? */
function inBox(ref: React.RefObject<HTMLElement | null>, cursor: Cursor) {
  const el = ref.current;
  if (!el) return false;
  const box = el.getBoundingClientRect();
  return (
    cursor.x >= box.left &&
    cursor.x <= box.right &&
    cursor.y >= box.top &&
    cursor.y <= box.bottom
  );
}

/** Cursor over a bubble that actually overflows? */
function overScrollingBubble(
  ref: React.RefObject<HTMLDivElement | null>,
  cursor: Cursor,
) {
  const el = ref.current;
  if (!el || el.scrollHeight <= el.clientHeight) return false;
  return inBox(ref, cursor);
}

const initialPosition = () => ({
  x: Math.max(0, window.innerWidth - CELL - MARGIN_X),
  y: Math.max(0, window.innerHeight - CELL - MARGIN_Y),
});

export function usePetMotion(
  canPlay: boolean,
  onWake: () => void,
  wakeOnHover = false,
) {
  const rootRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const eyesRef = useRef<HTMLDivElement>(null);
  const bubbleRef = useRef<HTMLDivElement>(null);
  // Open panel always claims the cursor.
  const panelRef = useRef<HTMLDivElement>(null);
  const [reaction, setReaction] = useState<Reaction>(null);
  const [quiet, setQuietState] = useState<Side | null>(null);
  const [petBurst, setPetBurst] = useState(0);
  const [earTwitch, setEarTwitch] = useState(false);
  const canPlayRef = useRef(canPlay);
  const wakeRef = useRef(onWake);
  const hoverWakeArmed = useRef(wakeOnHover);
  const interactive = useRef(false);
  const setHitTest = useRef<(on: boolean) => void>(() => {});
  const ignoreQueue = useRef(Promise.resolve());
  const switchingMonitor = useRef(false);
  const start = initialPosition();
  const motion = useRef<Motion>({
    x: spring(start.x),
    y: spring(start.y),
    sx: spring(1),
    sy: spring(1),
    angle: spring(0),
    cursor: { x: -1, y: -1, at: 0, vx: 0, vy: 0, speed: 0 },
    initialized: false,
    dragging: false,
    dragOffset: { x: 0, y: 0 },
    dragAt: 0,
    dragCursor: { x: 0, y: 0 },
    dragSpeed: 0,
    dragVx: 0,
    dragSign: 0,
    shakeTurns: [],
    angryUntil: 0,
    petUntil: 0,
    petEnergy: 0,
    petSign: 0,
    huntUntil: 0,
    huntSign: 0,
    huntTurns: [],
    huntTravel: 0,
    huntLastFastAt: 0,
    landUntil: 0,
    reaction: null,
    quiet: null,
    home: null,
  });

  useEffect(() => {
    canPlayRef.current = canPlay;
  }, [canPlay]);
  useEffect(() => {
    wakeRef.current = onWake;
  }, [onWake]);
  useEffect(() => {
    hoverWakeArmed.current = wakeOnHover;
  }, [wakeOnHover]);

  /** Tuck behind an edge or bring him back (ref + state). */
  const setQuiet = useCallback((side: Side | null) => {
    const m = motion.current;
    if (m.quiet === side) return;
    if (side !== null) {
      // Remembered so Come back returns him where he was standing.
      if (m.quiet === null) m.home = { x: m.x.target, y: m.y.target };
      m.x.target = peekX(side, window.innerWidth);
    } else if (!m.dragging && m.home) {
      // Skip mid-drag; clamp home away from the tucked edge.
      m.x.target = clamp(
        m.home.x,
        MARGIN_X,
        Math.max(MARGIN_X, window.innerWidth - CELL - MARGIN_X),
      );
      m.y.target = m.home.y;
    }
    m.quiet = side;
    setQuietState(side);
  }, []);

  /** What the menu item does: whichever edge he is nearer, or back out. */
  const toggleQuiet = useCallback(() => {
    const m = motion.current;
    if (m.quiet !== null) {
      setQuiet(null);
      return;
    }
    const centre = m.x.target + CELL / 2;
    setQuiet(centre < window.innerWidth / 2 ? "left" : "right");
  }, [setQuiet]);

  const publishReaction = useCallback((next: Reaction) => {
    const m = motion.current;
    if (m.reaction === next) return;
    m.reaction = next;
    setReaction(next);
  }, []);

  useEffect(() => {
    setEarTwitch(false);
    if (
      !canPlay ||
      quiet !== null ||
      reaction ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      return;
    }

    const timers: number[] = [];
    let disposed = false;
    const later = (fn: () => void, delay: number) => {
      const timer = window.setTimeout(() => {
        if (!disposed) fn();
      }, delay);
      timers.push(timer);
    };
    const schedule = () => {
      later(() => {
        setEarTwitch(true);
        later(() => setEarTwitch(false), 120);
        later(() => setEarTwitch(true), 220);
        later(() => {
          setEarTwitch(false);
          schedule();
        }, 340);
      }, 7_000 + Math.random() * 7_000);
    };
    schedule();

    return () => {
      disposed = true;
      timers.forEach(clearTimeout);
    };
  }, [canPlay, quiet, reaction]);

  useEffect(() => {
    const win = getCurrentWindow();
    const chooseHitTest = (on: boolean) => {
      if (interactive.current === on) return;
      interactive.current = on;
      ignoreQueue.current = ignoreQueue.current
        .then(() => win.setIgnoreCursorEvents(!on))
        .catch((error) => console.error("[mellow] hit test failed:", error));
    };
    setHitTest.current = chooseHitTest;

    const stop = listen<Cursor>("cursor", ({ payload }) => {
      const m = motion.current;
      const now = performance.now();
      const dt = m.cursor.at ? Math.max((now - m.cursor.at) / 1000, 1 / 240) : 0;
      const dx = payload.x - m.cursor.x;
      const dy = payload.y - m.cursor.y;
      const speed = dt ? Math.hypot(dx, dy) / dt : 0;
      m.cursor = {
        ...payload,
        at: now,
        vx: dt ? dx / dt : 0,
        vy: dt ? dy / dt : 0,
        speed,
      };

      // Cross-monitor only while physically dragging.
      if (
        m.dragging &&
        !switchingMonitor.current &&
        (payload.x < 0 || payload.x >= window.innerWidth ||
          payload.y < 0 || payload.y >= window.innerHeight)
      ) {
        switchingMonitor.current = true;
        void invoke("move_pet_to_cursor_monitor")
          .catch((error) => console.error("[mellow] monitor drag failed:", error))
          .finally(() => {
            window.setTimeout(() => { switchingMonitor.current = false; }, 100);
          });
      }

      // Keep drag attached via system cursor at monitor edges.
      if (m.dragging) {
        m.x.target = clamp(
          payload.x - m.dragOffset.x,
          -CELL * 0.35,
          window.innerWidth - CELL * 0.65,
        );
        m.y.target = clamp(
          payload.y - m.dragOffset.y,
          -BODY.y,
          window.innerHeight - sprites.ground * SCALE,
        );
      }

      // Hit-test hysteresis: tight enter, larger exit.
      const pad = interactive.current ? 20 : 7;
      // Quiet uses peekBox; sprites.hit is for standing poses.
      let inside: boolean;
      if (m.quiet !== null) {
        const box = peekBox(m.quiet, m.x.value, m.y.value);
        inside =
          payload.x >= box.x0 &&
          payload.x <= box.x1 &&
          payload.y >= box.y0 &&
          payload.y <= box.y1;
      } else {
        inside = hitsPet(payload, m.x.value, m.y.value, pad);
      }
      chooseHitTest(
        m.dragging ||
          inside ||
          overScrollingBubble(bubbleRef, payload) ||
          inBox(panelRef, payload),
      );

      // No hover-wake while tucked away.
      if (inside && hoverWakeArmed.current && m.quiet === null) {
        // Hover wake is local nap only (hotkey owns PTT).
        hoverWakeArmed.current = false;
        wakeRef.current();
        return;
      }

      // Quiet is quiet: no petting hearts, no hunting, no ear twitches.
      if (m.dragging || !canPlayRef.current || m.quiet !== null) return;

      const inHead =
        payload.x >= m.x.value + BODY.x &&
        payload.x <= m.x.value + BODY.x + BODY.w * 0.72 &&
        payload.y >= m.y.value + BODY.y &&
        payload.y <= m.y.value + BODY.y + BODY.h * 0.55;
      if (inHead && speed > 25 && speed < 550) {
        const sign = Math.sign(dx);
        if (sign && m.petSign && sign !== m.petSign) m.petEnergy += 24;
        m.petSign = sign || m.petSign;
        m.petEnergy += Math.min(Math.hypot(dx, dy), 18);
        if (m.petEnergy > 105) {
          m.petEnergy = 0;
          m.petUntil = now + 1_250;
          setPetBurst((burst) => burst + 1);
          wakeRef.current();
        }
      } else {
        m.petEnergy = Math.max(0, m.petEnergy - 8);
      }

      const near =
        payload.x >= m.x.value + BODY.x - 90 &&
        payload.x <= m.x.value + BODY.x + BODY.w + 90 &&
        payload.y >= m.y.value + BODY.y - 90 &&
        payload.y <= m.y.value + BODY.y + BODY.h + 90;
      const horizontalFast =
        Math.abs(m.cursor.vx) > 850 &&
        Math.abs(m.cursor.vx) > Math.abs(m.cursor.vy) * 2 &&
        Math.abs(dx) >= 8;

      if (!near) {
        m.huntSign = 0;
        m.huntTurns = [];
        m.huntTravel = 0;
        m.huntLastFastAt = 0;
      } else if (horizontalFast && now > m.petUntil) {
        // 160ms grace so left-right-left counts as hunt turns.
        if (m.huntLastFastAt && now - m.huntLastFastAt > 160) {
          m.huntSign = 0;
          m.huntTurns = [];
          m.huntTravel = 0;
        }
        const sign = Math.sign(m.cursor.vx);
        if (m.huntSign && sign !== m.huntSign) m.huntTurns.push(now);
        m.huntSign = sign;
        m.huntTravel += Math.abs(dx);
        m.huntLastFastAt = now;
        m.huntTurns = m.huntTurns.filter((at) => now - at <= 550);

        if (m.huntTurns.length >= 2 && m.huntTravel >= 80) {
          // Hunt is expression/gaze only; drag owns root x/y.
          m.huntUntil = now + 650;
          m.huntSign = 0;
          m.huntTurns = [];
          m.huntTravel = 0;
          wakeRef.current();
        }
      } else if (m.huntLastFastAt && now - m.huntLastFastAt > 160) {
        m.huntSign = 0;
        m.huntTurns = [];
        m.huntTravel = 0;
        m.huntLastFastAt = 0;
      }
    });

    return () => {
      stop.then((off) => off()).catch(() => {});
      setHitTest.current = () => {};
      ignoreQueue.current = ignoreQueue.current
        .then(() => win.setIgnoreCursorEvents(true))
        .catch(() => {});
    };
  }, []);

  useEffect(() => {
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const m = motion.current;
      const root = rootRef.current;
      const body = bodyRef.current;
      const eyes = eyesRef.current;
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;

      if (!m.initialized) {
        const p = initialPosition();
        m.x.value = m.x.target = p.x;
        m.y.value = m.y.target = p.y;
        m.initialized = true;
      }

      let next: Reaction = null;
      if (m.dragging) next = now < m.angryUntil ? "angry" : "drag";
      else if (now < m.petUntil) next = "pet";
      else if (now < m.huntUntil) next = "hunt";
      else if (now < m.angryUntil) next = "angry";
      publishReaction(next);

      if (m.dragging) {
        const stretch = clamp(m.dragSpeed / 2_800, 0, 0.45);
        m.sy.target = 1 + stretch;
        m.sx.target = 1 / (1 + stretch);
        m.angle.target = clamp(m.dragVx / 85, -11, 11);
      } else if (now < m.landUntil) {
        m.sx.target = 1.2;
        m.sy.target = 0.82;
        m.angle.target = 0;
      } else {
        m.sx.target = 1;
        m.sy.target = 1;
        // No extra lean — peek art is already leaning.
        m.angle.target =
          next === "angry"
            ? Math.sin(now / 38) * 8
            : next === "pet"
              ? Math.sin(now / 85) * 3
              : 0;
      }

      stepSpring(m.x, dt, m.dragging ? 12 : 7, 0.9);
      stepSpring(m.y, dt, m.dragging ? 12 : 7, 0.9);
      stepSpring(m.sx, dt, 8, 0.78);
      stepSpring(m.sy, dt, 8, 0.78);
      stepSpring(m.angle, dt, 9, 0.72);

      if (root) {
        root.style.transform = `translate3d(${Math.round(m.x.value)}px, ${Math.round(m.y.value)}px, 0)`;
      }
      if (body) {
        const sx = Math.round(m.sx.value * 64) / 64;
        const sy = Math.round(m.sy.value * 64) / 64;
        body.style.transform = `scale(${sx}, ${sy}) rotate(${Math.round(m.angle.value * 10) / 10}deg)`;
      }
      if (eyes) {
        const cx = m.x.value + CELL / 2;
        const cy = m.y.value + BODY.y + BODY.h * 0.4;
        eyes.style.setProperty(
          "--gaze-x",
          `${Math.round(clamp((m.cursor.x - cx) / 120, -1, 1)) * SCALE}px`,
        );
        eyes.style.setProperty(
          "--gaze-y",
          `${Math.round(clamp((m.cursor.y - cy) / 150, -1, 1)) * SCALE}px`,
        );
      }

      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [publishReaction]);

  useEffect(() => {
    const resize = () => {
      const m = motion.current;
      // Re-derive quiet X on resize (don't clamp him back).
      m.x.target =
        m.quiet !== null
          ? peekX(m.quiet, window.innerWidth)
          : clamp(m.x.target, -CELL * 0.35, window.innerWidth - CELL * 0.65);
      m.y.target = clamp(m.y.target, 0, window.innerHeight - CELL);
    };
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      const m = motion.current;
      event.currentTarget.setPointerCapture(event.pointerId);
      m.dragging = true;
      m.dragOffset = { x: event.clientX - m.x.value, y: event.clientY - m.y.value };
      m.dragAt = performance.now();
      m.dragCursor = { x: event.clientX, y: event.clientY };
      m.dragSpeed = 0;
      m.dragVx = 0;
      m.dragSign = 0;
      m.shakeTurns = [];
      setHitTest.current(true);
      onWake();
    },
    [onWake],
  );

  const onPointerMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const m = motion.current;
    if (!m.dragging) return;
    const now = performance.now();
    const dt = Math.max((now - m.dragAt) / 1000, 1 / 240);
    const vx = (event.clientX - m.dragCursor.x) / dt;
    const vy = (event.clientY - m.dragCursor.y) / dt;
    const sign = Math.abs(vx) > 320 ? Math.sign(vx) : 0;
    if (sign && m.dragSign && sign !== m.dragSign) {
      m.shakeTurns.push(now);
      m.shakeTurns = m.shakeTurns.filter((at) => now - at < 720);
      if (m.shakeTurns.length >= 3) {
        m.angryUntil = now + 1_250;
        m.shakeTurns = [];
      }
    }
    // Come back on first drag pixel, not on release.
    if (m.quiet !== null) setQuiet(null);
    if (sign) m.dragSign = sign;
    m.dragVx = vx;
    m.dragSpeed = Math.hypot(vx, vy);
    m.dragAt = now;
    m.dragCursor = { x: event.clientX, y: event.clientY };
    m.x.target = clamp(
      event.clientX - m.dragOffset.x,
      -CELL * 0.35,
      window.innerWidth - CELL * 0.65,
    );
    m.y.target = clamp(
      event.clientY - m.dragOffset.y,
      -BODY.y,
      window.innerHeight - sprites.ground * SCALE,
    );
  }, [setQuiet]);

  const release = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const m = motion.current;
      if (!m.dragging) return;
      m.dragging = false;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      // Drop into an edge to go quiet; no squash-land.
      const side = edgeAt(m.x.target, window.innerWidth);
      if (side) {
        setQuiet(side);
        return;
      }
      m.landUntil = performance.now() + 125;
      m.sx.velocity += 2.2;
      m.sy.velocity -= 2.2;
      m.angle.target = 0;
    },
    [setQuiet],
  );

  // Hand input back to whatever is under the overlay. The `cursor` stream only
  // fires on physical movement, so a hand-off to another window has to say so
  // itself rather than wait for the next mouse move.
  const releaseOverlay = useCallback(() => setHitTest.current(false), []);

  return {
    rootRef,
    bodyRef,
    eyesRef,
    bubbleRef,
    panelRef,
    releaseOverlay,
    reaction,
    quiet,
    setQuiet,
    toggleQuiet,
    petBurst,
    earTwitch,
    pointer: {
      onPointerDown,
      onPointerMove,
      onPointerUp: release,
      onPointerCancel: release,
    },
  };
}
