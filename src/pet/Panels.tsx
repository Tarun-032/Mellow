import { emit } from "@tauri-apps/api/event";
import { useCallback, useEffect, useState } from "react";
import {
  LIMITS,
  PHASE_LABEL,
  SETTING_LABEL,
  mmss,
  type Setting,
  type usePomodoro,
} from "./usePomodoro";

// Same HTTP host as Settings for reminder CRUD.
const API = "http://127.0.0.1:8765";

export type Reminder = {
  id: string;
  time: string;
  text: string;
  daily: boolean;
  last_fired: string;
};

type Draft = { id?: string; time: string; text: string; daily: boolean };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`mellowd said ${res.status}`);
  return (await res.json()) as T;
}

const hhmm = (date: Date) =>
  `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;

/** Panel frame using the speech-bubble 9-slice. */
function Panel({
  title,
  actions,
  children,
}: {
  title: string;
  actions: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="panel">
      <div className="panel__bar">
        <span className="panel__title">{title}</span>
        <span className="panel__actions">{actions}</span>
      </div>
      {children}
    </div>
  );
}

/** Stepper for one settable number (buttons, not an input). */
function Stepper({
  name,
  value,
  onStep,
}: {
  name: Setting;
  value: number;
  onStep: (key: Setting, step: number) => void;
}) {
  const [low, high, step] = LIMITS[name];
  return (
    <div className="panel__row panel__setting">
      <span className="panel__opt">{SETTING_LABEL[name]}</span>
      <button
        className="panel__btn panel__btn--step"
        onClick={() => onStep(name, -step)}
        disabled={value <= low}
        aria-label={`less ${name}`}
      >
        &minus;
      </button>
      <span className="panel__num">{value}</span>
      <button
        className="panel__btn panel__btn--step"
        onClick={() => onStep(name, step)}
        disabled={value >= high}
        aria-label={`more ${name}`}
      >
        +
      </button>
    </div>
  );
}

export function PomodoroPanel({
  timer,
  onClose,
}: {
  timer: ReturnType<typeof usePomodoro>;
  onClose: () => void;
}) {
  const idle = !timer.running && !timer.paused;
  // Spell the phase so a bare clock doesn't look running.
  const status = idle
    ? "Ready"
    : timer.paused
      ? "Paused"
      : PHASE_LABEL[timer.phase];
  const tone =
    idle || timer.paused
      ? "idle"
      : timer.phase === "focus"
        ? "focus"
        : "break";
  return (
    <Panel
      title="Pomodoro"
      actions={
        <button className="panel__btn" onClick={onClose}>
          Close
        </button>
      }
    >
      <div className={`panel__clock panel__clock--${tone}`}>
        <span className="panel__phase">{status}</span>
        <span className="panel__count">{mmss(timer.remaining)}</span>
      </div>
      {idle ? (
        // Steppers replace the idle subtitle.
        <div className="panel__settings">
          {(Object.keys(LIMITS) as Setting[]).map((name) => (
            <Stepper
              key={name}
              name={name}
              value={timer.settings[name]}
              onStep={timer.adjust}
            />
          ))}
        </div>
      ) : (
        <p className="panel__round">
          round {timer.round} of {timer.rounds}
        </p>
      )}
      <div className="panel__row panel__row--buttons">
        {idle ? (
          <button className="panel__btn panel__btn--wide" onClick={() => timer.start()}>
            Start
          </button>
        ) : (
          <>
            <button
              className="panel__btn panel__btn--wide"
              onClick={timer.paused ? timer.resume : timer.pause}
            >
              {timer.paused ? "Resume" : "Pause"}
            </button>
            <button className="panel__btn panel__btn--wide" onClick={timer.stop}>
              Stop
            </button>
          </>
        )}
      </div>
    </Panel>
  );
}

export function ReminderPanel({ onClose }: { onClose: () => void }) {
  // null = first fetch in flight (vs empty list).
  const [items, setItems] = useState<Reminder[] | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [error, setError] = useState("");
  const typing = draft !== null;

  useEffect(() => {
    request<{ reminders: Reminder[] }>("/reminders")
      .then((data) => setItems(data.reminders))
      .catch(() => {
        setItems([]);
        setError("can't reach mellowd");
      });
  }, []);

  // Borrow keyboard focus only while a field needs it.
  useEffect(() => {
    emit("pet-focus", { focus: typing }).catch(() => {});
  }, [typing]);
  // Always return focus when the panel closes.
  useEffect(
    () => () => {
      emit("pet-focus", { focus: false }).catch(() => {});
    },
    [],
  );

  const persist = useCallback(async (next: Draft[]) => {
    try {
      const data = await request<{ reminders: Reminder[] }>("/reminders", {
        method: "PUT",
        body: JSON.stringify({ reminders: next }),
      });
      // Show the sidecar's normalised list as ground truth.
      setItems(data.reminders);
      setError("");
      return true;
    } catch {
      setError("couldn't save that");
      return false;
    }
  }, []);

  const save = async () => {
    if (!draft || !draft.text.trim()) return;
    const rest = (items ?? []).filter((item) => item.id !== draft.id);
    const ok = await persist(
      [...rest, draft].sort((a, b) => a.time.localeCompare(b.time)),
    );
    // Close after save — storage is the confirmation.
    if (ok) onClose();
  };

  const remove = (id: string) =>
    persist((items ?? []).filter((item) => item.id !== id));

  return (
    <Panel
      title="Reminder"
      actions={
        <>
          {!typing && (
            <button
              className="panel__btn"
              onClick={() =>
                setDraft({ time: hhmm(new Date()), text: "", daily: false })
              }
            >
              Add
            </button>
          )}
          <button className="panel__btn" onClick={onClose}>
            Close
          </button>
        </>
      }
    >
      {items === null ? (
        <p className="panel__hint">one moment...</p>
      ) : items.length === 0 && !typing ? (
        <p className="panel__hint">
          Add a reminder and Mellow will tell you on time.
        </p>
      ) : (
        <ul className="panel__list">
          {items.map((item) => (
            <li className="panel__row" key={item.id}>
              <span className="panel__time">{item.time}</span>
              <span className="panel__what">{item.text}</span>
              <span className="panel__when">{item.daily ? "Daily" : "Once"}</span>
              <button
                className="panel__btn"
                onClick={() =>
                  setDraft({
                    id: item.id,
                    time: item.time,
                    text: item.text,
                    daily: item.daily,
                  })
                }
              >
                Edit
              </button>
              <button
                className="panel__btn"
                onClick={() => remove(item.id)}
                aria-label={`delete ${item.text}`}
              >
                &times;
              </button>
            </li>
          ))}
        </ul>
      )}

      {draft && (
        <div className="panel__form">
          <div className="panel__row">
            <input
              className="panel__input panel__input--time"
              type="time"
              value={draft.time}
              onChange={(e) => setDraft({ ...draft, time: e.target.value })}
            />
            <button
              className={`panel__btn${draft.daily ? " panel__btn--ghost" : ""}`}
              onClick={() => setDraft({ ...draft, daily: false })}
            >
              Once
            </button>
            <button
              className={`panel__btn${draft.daily ? "" : " panel__btn--ghost"}`}
              onClick={() => setDraft({ ...draft, daily: true })}
            >
              Daily
            </button>
          </div>
          <div className="panel__row">
            <input
              className="panel__input"
              autoFocus
              maxLength={200}
              placeholder="What should Mellow remind you?"
              value={draft.text}
              onChange={(e) => setDraft({ ...draft, text: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === "Enter") save();
                if (e.key === "Escape") setDraft(null);
              }}
            />
          </div>
          <div className="panel__row panel__row--buttons">
            <button
              className="panel__btn panel__btn--wide"
              onClick={() => setDraft(null)}
            >
              Cancel
            </button>
            <button
              className="panel__btn panel__btn--wide"
              onClick={save}
              disabled={!draft.text.trim()}
            >
              Save
            </button>
          </div>
        </div>
      )}
      {error && <p className="panel__hint panel__hint--error">{error}</p>}
    </Panel>
  );
}
