import { useCallback, useEffect, useRef, useState } from "react";

export type Phase = "focus" | "break";
export type Setting = "focus" | "break" | "rounds";
export type Settings = Record<Setting, number>;

/** [min, max, step] per setting. */
export const LIMITS: Record<Setting, [number, number, number]> = {
  focus: [5, 60, 5],
  break: [1, 30, 1],
  rounds: [1, 8, 1],
};

export const SETTING_LABEL: Record<Setting, string> = {
  focus: "Focus",
  break: "Break",
  rounds: "Rounds",
};

const DEFAULTS: Settings = { focus: 25, break: 5, rounds: 4 };
const KEY = "mellow.pomodoro";

export const PHASE_LABEL: Record<Phase, string> = {
  focus: "Focus",
  break: "Break",
};

/** Coerce stored settings; out-of-range -> default, not clamp. */
export function clampSettings(raw: unknown): Settings {
  const source = (raw ?? {}) as Partial<Record<Setting, unknown>>;
  const out = { ...DEFAULTS };
  for (const key of Object.keys(DEFAULTS) as Setting[]) {
    const [low, high] = LIMITS[key];
    const value = source[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      const whole = Math.round(value);
      if (whole >= low && whole <= high) out[key] = whole;
    }
  }
  return out;
}

/** localStorage key for pomodoro lengths (not config.json). */
const load = (): Settings => {
  try {
    return clampSettings(JSON.parse(localStorage.getItem(KEY) ?? "null"));
  } catch {
    return { ...DEFAULTS };
  }
};

const save = (settings: Settings) => {
  try {
    localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    // Ignore localStorage write failures.
  }
};

const secondsFor = (phase: Phase, settings: Settings) =>
  (phase === "focus" ? settings.focus : settings.break) * 60;

/** Next phase after the current one, or null when done. */
export function advance(
  phase: Phase,
  round: number,
  rounds: number,
): { phase: Phase; round: number } | null {
  // The break after the final focus round has no job: the session is done.
  if (phase === "focus") {
    return round >= rounds ? null : { phase: "break", round };
  }
  return { phase: "focus", round: round + 1 };
}

/** mm:ss with padded minutes and seconds. */
export const mmss = (seconds: number) => {
  const safe = Math.max(0, seconds);
  const m = Math.floor(safe / 60);
  const s = safe % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
};

/** Frontend-only pomodoro (sidecar has no opinion). */
export function usePomodoro(onFire: (text: string) => void) {
  const [settings, setSettings] = useState<Settings>(load);
  const [phase, setPhase] = useState<Phase>("focus");
  const [round, setRound] = useState(1);
  // Wall-clock deadline (not a decrementing counter).
  const [deadline, setDeadline] = useState<number | null>(null);
  const [remaining, setRemaining] = useState(() => load().focus * 60);
  const [paused, setPaused] = useState(false);
  // onFire ref so parent callback churn won't reset the interval.
  const fire = useRef(onFire);

  useEffect(() => {
    fire.current = onFire;
  }, [onFire]);

  const idle = deadline === null && !paused;

  // Idle clock previews the focus stepper value.
  useEffect(() => {
    if (idle) setRemaining(settings.focus * 60);
  }, [idle, settings.focus]);

  useEffect(() => {
    if (deadline === null) return;

    const tick = () => {
      const left = Math.round((deadline - Date.now()) / 1000);
      if (left > 0) {
        setRemaining(left);
        return;
      }
      const next = advance(phase, round, settings.rounds);
      if (next === null) {
        setDeadline(null);
        setPhase("focus");
        setRound(1);
        setRemaining(settings.focus * 60);
        fire.current("That's the session done. Nice work.");
        return;
      }
      // Advance to the next phase immediately at boundaries.
      const length = secondsFor(next.phase, settings);
      setPhase(next.phase);
      setRound(next.round);
      setRemaining(length);
      setDeadline(Date.now() + length * 1000);
      fire.current(
        next.phase === "focus"
          ? "Break's over. Back to it."
          : "That's the round done. Go and stretch.",
      );
    };

    tick();
    // Tick 4Hz so the displayed second stays current.
    const timer = setInterval(tick, 250);
    return () => clearInterval(timer);
  }, [deadline, phase, round, settings]);

  /** Nudge one setting by one step (idle only). */
  const adjust = useCallback((key: Setting, step: number) => {
    setSettings((current) => {
      const [low, high] = LIMITS[key];
      const value = Math.min(high, Math.max(low, current[key] + step));
      const next = { ...current, [key]: value };
      save(next);
      return next;
    });
  }, []);

  /** Start a round; optional focusFor override. */
  const start = useCallback(
    (focusFor?: number) => {
      const [low, high] = LIMITS.focus;
      const chosen =
        typeof focusFor === "number" && Number.isFinite(focusFor)
          ? Math.min(high, Math.max(low, Math.round(focusFor)))
          : settings.focus;
      if (chosen !== settings.focus) {
        setSettings((current) => {
          const next = { ...current, focus: chosen };
          save(next);
          return next;
        });
      }
      const length = chosen * 60;
      setPhase("focus");
      setRound(1);
      setRemaining(length);
      setPaused(false);
      setDeadline(Date.now() + length * 1000);
    },
    [settings.focus],
  );

  const pause = useCallback(() => {
    setPaused(true);
    setDeadline(null);
  }, []);

  const resume = useCallback(() => {
    setPaused(false);
    setDeadline(Date.now() + remaining * 1000);
  }, [remaining]);

  const stop = useCallback(() => {
    setDeadline(null);
    setPaused(false);
    setPhase("focus");
    setRound(1);
    setRemaining(settings.focus * 60);
  }, [settings.focus]);

  return {
    settings,
    adjust,
    phase,
    round,
    rounds: settings.rounds,
    remaining,
    running: deadline !== null,
    paused,
    start,
    pause,
    resume,
    stop,
  };
}
