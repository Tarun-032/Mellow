import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { emit, emitTo, listen } from "@tauri-apps/api/event";
import { PomodoroPanel, ReminderPanel } from "./Panels";
import { MeetingPanel } from "../meetings/MeetingPanel";
import { clock as meetingClock, useMeeting, viewMeeting } from "../meetings/useMeeting";
import { GUIDE_DIALOGUE_KEY, type GuideDialogue } from "./guideDialogue";
import { PHASE_LABEL, mmss, usePomodoro } from "./usePomodoro";
import { useSocket } from "./useSocket";
import { bonePlacement, usePetMotion, type Reaction } from "./usePetMotion";
import "./sprites.css"; // generated: --cell-<state> indices into sprites.png
import "./pet.css";

const YAWN_AFTER = 60_000;
// Reading time after the turn ends (not while still speaking).
const DISMISS_AFTER = 20_000;
// Reading time for a pointed reply when no audio was played.
const POINT_DISMISS = 10_000;
// Yawn sequence length before sleep.
const YAWN_LENGTH = 3_200;
// Settle delay at break start (uses the yawn, then sleeps).
const DOZE_AFTER = 1_500;
const PARTICLES = [0, 1, 2, 3, 4];
// Cap on unanswered alert attention.
const ALERT_CAP = 120_000;

/** Badge wording per meeting state; the panel spells things out at length. */
const MEETING_LABEL: Record<string, string> = {
  starting: "Getting ready",
  recording: "Notes",
  paused: "Paused",
  finalizing: "Saving notes",
};

/** Which panel is hovering over Mellow, if any. */
type Panel = "pomodoro" | "reminders" | "meeting" | null;

/** Nap clock for the mic; pose may be overruled by a pomodoro. */
type Nap = "awake" | "yawn" | "sleeping";

/** Hold overrides nap: awake, sleep, or null for the clock. */
type Hold = "awake" | "sleep" | null;

type GuideAck = { accepted: boolean; arrived: boolean };

// Monotonic guide revisions; Date.now beats a remounted window's stale cmds.
let lastGuideRevision = 0;
function nextGuideRevision() {
  lastGuideRevision = Math.max(lastGuideRevision + 1, Date.now() * 1_000);
  return lastGuideRevision;
}

function resolvePose(
  state: "idle" | "listening" | "thinking" | "looking" | "talking",
  nap: Nap,
  reaction: Reaction,
  alerting: boolean,
  quiet: boolean,
) {
  // Physical reactions outrank sidecar state.
  if (reaction === "angry") return "angry";
  if (reaction === "drag") return "listening";
  if (reaction === "pet") return "petting";
  if (reaction === "hunt") return "hunt";
  // Quiet above alert, below reactions.
  if (quiet) return "peek";
  // Fired timer outranks sidecar until acknowledged.
  if (alerting) return "alert";
  // Looking uses thinking pose (not alert).
  if (state === "looking") return "thinking";
  if (state !== "idle") return state;
  return nap === "awake" ? "idle" : nap;
}

export default function Pet() {
  const {
    connected,
    state,
    microphone,
    transcript,
    reply,
    error,
    speak,
    reminder,
    point,
    // Socket timer request; hook below owns the live round as `timer`.
    timer: asked,
    send,
    clear,
    dismissReminder,
  } = useSocket();
  const meeting = useMeeting();
  const meetingActive = Boolean(meeting.status?.active);
  const completedMeetingId = meeting.status?.status === "complete" ? meeting.status.id : null;
  const [dismissedMeetingId, setDismissedMeetingId] = useState<string | null>(null);
  const meetingSaved = Boolean(completedMeetingId && completedMeetingId !== dismissedMeetingId);
  useEffect(() => {
    if (!completedMeetingId || completedMeetingId === dismissedMeetingId) return;
    const timeout = window.setTimeout(() => setDismissedMeetingId(completedMeetingId), 15_000);
    return () => window.clearTimeout(timeout);
  }, [completedMeetingId, dismissedMeetingId]);
  const [nap, setNap] = useState<Nap>("awake");
  const [panel, setPanel] = useState<Panel>(null);
  // Local pomodoro fire (separate from sidecar reminders).
  const [fired, setFired] = useState("");
  // Queued while quiet; sidecar already deleted them from disk.
  const [waiting, setWaiting] = useState<string[]>([]);
  const timer = usePomodoro(setFired);
  const alert = fired || reminder;
  useEffect(() => { if (meetingActive && alert) setPanel("meeting"); }, [meetingActive, alert]);
  // Hold awake through sidecar work; break sleep; otherwise use the nap clock.
  const holdMode: Hold =
    meetingActive || panel !== null || alert !== "" || state !== "idle"
      ? "awake"
      : !timer.running
        ? null
        : timer.phase === "focus"
          ? "awake"
          : "sleep";
  const hold = useRef(holdMode);
  const naps = useRef<number[]>([]);
  // Outside the effect so StrictMode remounts stay in sync.
  const held = useRef(false);

  // Local inactivity (interaction with Mellow only).
  const wake = useCallback(() => {
    naps.current.forEach(clearTimeout);
    setNap("awake");
    // Always arm: mic follows `nap`, not the visible hold.
    const yawnAt = hold.current === "sleep" ? DOZE_AFTER : YAWN_AFTER;
    naps.current = [
      // Clear on yawn (not wake); skip while held awake.
      setTimeout(() => {
        setNap("yawn");
        if (hold.current !== "awake") clear();
      }, yawnAt),
      setTimeout(() => setNap("sleeping"), yawnAt + YAWN_LENGTH),
    ];
  }, [clear]);

  // Wake existing renderer after onboarding (it may already be asleep).
  useEffect(() => {
    const stop = listen("pet-wake", wake);
    return () => {
      stop.then((off) => off()).catch(() => {});
    };
  }, [wake]);

  // Sidecar activity counts as interaction.
  useEffect(() => {
    wake();
  }, [state, wake]);

  useEffect(() => () => naps.current.forEach(clearTimeout), []);

  // Re-arm naps whenever hold mode changes.
  useEffect(() => {
    hold.current = holdMode;
    wake();
  }, [holdMode, wake]);

  // Visible pose; mic still follows `nap`.
  const pose: Nap = holdMode === "awake" ? "awake" : nap;

  const dismiss = useCallback(() => {
    setFired("");
    dismissReminder();
  }, [dismissReminder]);

  // Auto-dismiss unanswered alerts.
  useEffect(() => {
    if (!alert) return;
    const timeout = setTimeout(dismiss, ALERT_CAP);
    return () => clearTimeout(timeout);
  }, [alert, dismiss]);

  // Spoken timer request: start and open the panel.
  useEffect(() => {
    if (!asked) return;
    if (asked.action === "stop") {
      timer.stop();
      return;
    }
    timer.start(asked.minutes ?? undefined);
    setPanel("pomodoro");
  }, [asked]);

  // Native menu opens panels (pet window is click-through).
  useEffect(() => {
    const stop = listen<string>("open-panel", ({ payload }) => {
      setPanel(payload === "meeting" ? "meeting" : payload === "reminders" ? "reminders" : "pomodoro");
    });
    return () => {
      stop.then((off) => off()).catch(() => {});
    };
  }, []);

  const motion = usePetMotion(
    !meetingActive && state === "idle" && pose === "awake" && !alert,
    wake,
    pose === "sleeping",
  );
  // Quiet edge from the motion hook.
  const { quiet, setQuiet, toggleQuiet } = motion;
  // Hand off to the settings window: drop the panel and stop eating clicks
  // first, or the new window opens under a full-screen overlay that has them.
  const viewMeetings = useCallback(
    (id: string | null) => {
      setDismissedMeetingId(id);
      setPanel(null);
      motion.releaseOverlay();
      void viewMeeting(id);
    },
    [motion],
  );

  // Settings can also be opened from the tray with a panel still up; the shell
  // drops click-through for us, so mirror it or the pet never re-arms.
  useEffect(() => {
    const stop = listen("pet-released", () => {
      setPanel(null);
      motion.releaseOverlay();
    });
    return () => {
      stop.then((off) => off()).catch(() => {});
    };
  }, [motion]);
  useEffect(() => {
    if (meetingActive || panel === "meeting") setQuiet(null);
    if (meetingActive) { held.current = false; clear(); }
  }, [meetingActive, panel, setQuiet, clear]);
  // Pointing: words ride the bone; skip if quiet or asleep.
  const pointing = point !== null && !quiet && pose === "awake";
  // Wait for native bone arrival before showing dialogue.
  const [landed, setLanded] = useState(false);
  // Remember a pointing turn after the bone clears.
  const pointed = useRef(false);
  // Audio completion, not elapsed time, dismisses spoken pointing turns.
  const spokeWhilePointing = useRef(false);
  const guideRevision = useRef(0);
  const dialogueRevision = useRef(0);

  useEffect(() => {
    const stop = listen<{ revision: number }>("guide-arrived", ({ payload }) => {
      if (payload.revision === guideRevision.current) setLanded(true);
    });
    return () => {
      stop.then((off) => off()).catch(() => {});
    };
  }, []);

  // Clear native guide state on unmount.
  useEffect(() => () => {
    dialogueRevision.current += 1;
    localStorage.removeItem(GUIDE_DIALOGUE_KEY);
    void invoke("guide_set_dialogue", { visible: false }).catch(() => {});
    void invoke("guide_clear", { revision: nextGuideRevision() }).catch(() => {});
  }, []);

  useEffect(() => {
    const revision = nextGuideRevision();
    guideRevision.current = revision;
    setLanded(false);

    if (!point) {
      void invoke<GuideAck>("guide_clear", { revision }).catch((error) =>
        console.error("[mellow] guide return failed:", error),
      );
      return;
    }

    pointed.current = true;
    void invoke<GuideAck>("guide_set_target", {
      revision,
      nx: point.nx,
      ny: point.ny,
      monitor: point.monitor,
    })
      .then((ack) => {
        if (revision === guideRevision.current && ack.arrived) setLanded(true);
      })
      .catch((error) => {
        console.error("[mellow] guide target failed:", error);
        // Show dialogue even if native guide failed.
        if (revision === guideRevision.current) setLanded(true);
      });
  }, [point]);

  useEffect(() => {
    void invoke("guide_set_quiet", { quiet: quiet !== null }).catch((error) =>
      console.error("[mellow] guide visibility failed:", error),
    );
  }, [quiet]);

  // Reset pointed flag when a new turn starts.
  useEffect(() => {
    if (state === "listening" || state === "thinking") {
      pointed.current = false;
      spokeWhilePointing.current = false;
    } else if (state === "talking" && pointed.current) {
      spokeWhilePointing.current = true;
    }
  }, [state, point]);

  // Queue firings while quiet (both channels, not combined alert).
  useEffect(() => {
    if (!quiet || (!fired && !reminder)) return;
    setWaiting((queue) => [...queue, fired, reminder].filter(Boolean));
    setFired("");
    dismissReminder();
  }, [quiet, fired, reminder, dismissReminder]);

  // Close panels when going quiet.
  useEffect(() => {
    if (quiet) setPanel(null);
  }, [quiet]);

  useEffect(() => {
    if (quiet || alert || waiting.length === 0) return;
    setFired(waiting[0]);
    setWaiting((queue) => queue.slice(1));
  }, [quiet, alert, waiting]);

  // React to the Rust-registered PTT hotkey.
  useEffect(() => {
    const stop = listen<boolean>("ptt", ({ payload: down }) => {
      if (meetingActive) return;
      // Ignore PTT while mic is warming (and matching releases).
      if (down && microphone === "warming") return;
      if (!down && !held.current) return;
      // Dedupe Windows key-repeat.
      if (down === held.current) return;
      held.current = down;
      if (down) {
        // Come back from quiet when talking.
        setQuiet(null);
        wake();
        // Clear stale bubble on press.
        clear();
        send({ type: "ptt_start" });
      } else {
        send({ type: "ptt_end" });
      }
    });
    return () => {
      stop.then((off) => off()).catch(() => {});
    };
  }, [send, wake, clear, setQuiet, microphone, meetingActive]);

  // Mic open while awake and not quiet.
  const listening = !meetingActive && nap !== "sleeping" && !quiet;
  useEffect(() => {
    if (connected) send({ type: "awake", value: listening });
  }, [connected, listening, send]);

  // Native context menu (DOM can't receive clicks on click-through).
  const openMenu = useCallback(
    (event: React.MouseEvent) => {
      // Suppress WebView2's default menu.
      event.preventDefault();
      wake();
      // Pass speak/quiet so Rust can label menu items.
      emit("pet-menu", { speak, quiet: quiet !== null, meeting: meetingActive }).catch(() => {});
    },
    [speak, wake, quiet, meetingActive],
  );

  useEffect(() => {
    const stop = listen("pet-quiet", () => { if (!meetingActive) toggleQuiet(); });
    return () => {
      stop.then((off) => off()).catch(() => {});
    };
  }, [toggleQuiet, meetingActive]);

  useEffect(() => {
    const stop = listen("toggle-speak", () => {
      send({ type: "set_speak", value: !speak });
    });
    return () => {
      stop.then((off) => off()).catch(() => {});
    };
  }, [send, speak]);

  // New chat: clear bubble and tell the sidecar.
  useEffect(() => {
    const stop = listen("new-chat", () => {
      clear();
      send({ type: "new_conversation" });
    });
    return () => {
      stop.then((off) => off()).catch(() => {});
    };
  }, [send, clear]);

  // Display priority: alert > error > reply > transcript.
  const said = alert || error || reply || transcript;

  // Auto-clear finished idle exchanges (own timer, not naps).
  useEffect(() => {
    // `idle` arrives only after queued speech has actually finished playing.
    if (state !== "idle" || !said || alert) return;
    if (pointed.current && spokeWhilePointing.current) {
      pointed.current = false;
      spokeWhilePointing.current = false;
      clear();
      return;
    }
    const reading = setTimeout(
      () => {
        pointed.current = false;
        spokeWhilePointing.current = false;
        clear();
      },
      pointing || pointed.current ? POINT_DISMISS : DISMISS_AFTER,
    );
    return () => clearTimeout(reading);
    // `point` restarts the clock per walkthrough step.
  }, [state, said, alert, pointing, point, clear]);

  const shown = meetingActive ? "writing" :
    motion.earTwitch &&
    state === "idle" &&
    pose === "awake" &&
    !alert &&
    !motion.reaction
      ? "ear"
      : resolvePose(state, pose, motion.reaction, alert !== "", quiet !== null);

  const spot = pointing
    ? bonePlacement(point, window.innerWidth, window.innerHeight)
    : null;
  const remotePointing = Boolean(pointing && point?.monitor);
  const remoteDialogue = Boolean(remotePointing && landed && said);

  useEffect(() => {
    const revision = ++dialogueRevision.current;
    const update = async () => {
      // Show native dialogue window before emitting text.
      const payload: GuideDialogue | null = remoteDialogue && spot
        ? {
            text: said,
            error: error !== "",
            side: spot.side,
            lift: spot.lift,
          }
        : null;
      if (payload) {
        // localStorage backup if the guide WebView reloads.
        localStorage.setItem(GUIDE_DIALOGUE_KEY, JSON.stringify(payload));
      } else {
        localStorage.removeItem(GUIDE_DIALOGUE_KEY);
      }
      await invoke("guide_set_dialogue", {
        visible: remoteDialogue,
        monitor: point?.monitor,
        nx: point?.nx,
        ny: point?.ny,
        side: spot?.side,
        lift: spot?.lift,
      });
      // Drop stale async dialogue updates.
      if (revision !== dialogueRevision.current) return;
      if (payload) {
        await emitTo("guide-bubble", "guide-dialogue", payload);
      }
    };
    void update().catch((error) =>
      console.error("[mellow] guide dialogue update failed:", error),
    );
  }, [remoteDialogue, point, said, error, spot?.side, spot?.lift]);

  return (
    <div className="stage">
      {/* Fallback bone for older sidecars without monitor id. */}
      {spot && !point?.monitor && (
        <div
          className="bone"
          style={{
            transform: `translate3d(${Math.round(spot.x)}px, ${Math.round(spot.y)}px, 0)`,
          }}
        >
          {said && landed && (
            <div className={`bubble bubble--bone is-${spot.side} is-${spot.lift}`}>
              <div
                className={`bubble__text${error ? " bubble__text--error" : ""}`}
              >
                {said}
              </div>
            </div>
          )}
        </div>
      )}
      <div
        className="pet-root"
        data-reaction={meetingActive ? "none" : motion.reaction ?? "none"}
        data-meeting={meeting.status?.status}
        data-quiet={quiet ?? undefined}
        ref={motion.rootRef}
      >
        {!meetingActive && connected && microphone === "warming" && !quiet && pose === "awake" && (
          <div
            className="mic-warmup"
            role="status"
            aria-label="Getting the microphone ready"
          >
            {Array.from({ length: 8 }, (_, dot) => <i key={dot} />)}
          </div>
        )}
        {/* Waiting marker while quiet (not a count). */}
        {quiet && waiting.length > 0 && <i className="quiet-dot" />}
        {!panel && (meetingActive || meetingSaved) && (
          <div className="badge badge--meeting" data-state={meeting.status?.status} ref={motion.panelRef}>
            <button
              type="button"
              className="badge__label"
              onClick={() => meetingSaved ? viewMeetings(meeting.status?.id ?? null) : setPanel("meeting")}
              title={meetingSaved ? "View notes in Settings" : "Open meeting controls"}
            >
              {meetingSaved ? "View notes" : `${MEETING_LABEL[meeting.status?.status ?? ""] ?? "Notes"} ${meetingClock(meeting.status?.duration || 0)}`}
            </button>
            {(meeting.status?.status === "recording" || meeting.status?.status === "paused") && (
              <>
                <button
                  type="button"
                  className="badge__icon"
                  onClick={() => void meeting.control(meeting.status?.status === "paused" ? "resume" : "pause")}
                  aria-label={meeting.status?.status === "paused" ? "Resume recording" : "Pause recording"}
                  title={meeting.status?.status === "paused" ? "Resume" : "Pause"}
                >
                  <svg viewBox="0 0 10 10" aria-hidden="true">
                    {meeting.status?.status === "paused" ? (
                      <polygon points="2,1 9,5 2,9" />
                    ) : (
                      <>
                        <rect x="2" y="1" width="2.5" height="8" />
                        <rect x="5.5" y="1" width="2.5" height="8" />
                      </>
                    )}
                  </svg>
                </button>
                <button
                  type="button"
                  className="badge__icon"
                  onClick={() => void meeting.control("stop")}
                  aria-label="Stop the meeting and save the transcript"
                  title="Stop & save"
                >
                  ×
                </button>
              </>
            )}
          </div>
        )}
        {!panel && !quiet && !meetingActive && !meetingSaved && (timer.running || timer.paused) && (
          <div
            className={`badge badge--${timer.phase}`}
          >
            {PHASE_LABEL[timer.phase]} {mmss(timer.remaining)}
          </div>
        )}
        {panel && (
          <div className="panel-anchor" ref={motion.panelRef}>
            {panel === "meeting" ? (
              <MeetingPanel status={meeting.status} connectionError={meeting.error} refresh={meeting.refresh} alert={alert} onDismissAlert={dismiss} onClose={() => setPanel(null)} onView={viewMeetings} />
            ) : panel === "pomodoro" ? (
              <PomodoroPanel timer={timer} onClose={() => setPanel(null)} />
            ) : (
              <ReminderPanel onClose={() => setPanel(null)} />
            )}
          </div>
        )}
        {/* Bubble only when awake, no panel, not pointing. */}
        {!panel && !quiet && !pointing && pose === "awake" &&
          // Skip empty listening balloon.
          (state === "thinking" || state === "looking" || said) && (
          <div className="bubble">
            {(state === "thinking" || state === "looking") && !said ? (
              <div
                className="bubble__dots"
                role="status"
                aria-label={state === "looking" ? "Looking at your screen" : "Thinking"}
              >
                <i />
                <i />
                <i />
              </div>
            ) : (
              <div
                className={`bubble__text${error ? " bubble__text--error" : ""}`}
                ref={motion.bubbleRef}
              >
                {said}
              </div>
            )}
          </div>
        )}
        {!meetingActive && motion.reaction === "pet" && (
          <div
            className="particles particles--hearts"
            key={`hearts-${motion.petBurst}`}
            aria-hidden="true"
          >
            {PARTICLES.map((particle) => <i key={particle} />)}
          </div>
        )}
        {!meetingActive && motion.reaction === "angry" && (
          <div className="particles particles--steam" aria-hidden="true">
            {PARTICLES.slice(0, 3).map((particle) => <i key={particle} />)}
          </div>
        )}
        {shown === "sleeping" && (
          <div className="particles particles--sleep" aria-hidden="true">
            {PARTICLES.slice(0, 3).map((particle) => <i key={particle}>Z</i>)}
          </div>
        )}
        <div
          className="pet-body"
          ref={motion.bodyRef}
          onContextMenu={openMenu}
          {...motion.pointer}
          onPointerDown={(event) => {
            dismiss();
            motion.pointer.onPointerDown(event);
          }}
        >
          <div className={`pet-art pet--${shown}`}>
            <div
              className={`pet-sprite${connected ? "" : " pet--offline"}`}
              role="img"
              aria-label={
                !connected
                  ? "Mellow is offline"
                  : microphone === "warming"
                    ? "Mellow is getting the microphone ready"
                    : "Mellow"
              }
            />
            <div
              className={`pet-eyes pet-eyes--${shown}`}
              ref={motion.eyesRef}
              aria-hidden="true"
            >
              <i className="pet-eye pet-eye--left" />
              <i className="pet-eye pet-eye--right" />
            </div>
            <i className={`pet-mouth pet-mouth--${shown}`} aria-hidden="true" />
          </div>
        </div>
      </div>
    </div>
  );
}
