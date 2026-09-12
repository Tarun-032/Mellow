import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { emit, emitTo, listen } from "@tauri-apps/api/event";
import { PomodoroPanel, ReminderPanel } from "./Panels";
import { MeetingPanel } from "../meetings/MeetingPanel";
import { clock as meetingClock, useMeeting, viewMeeting } from "../meetings/useMeeting";
import { GUIDE_DIALOGUE_KEY, type GuideDialogue } from "./guideDialogue";
import { PHASE_LABEL, mmss, usePomodoro } from "./usePomodoro";
import { useSocket } from "./useSocket";
import { useCoat } from "../ui/coatApply";
import { WritingPanel } from "./WritingPanel";
import { bonePlacement, usePetMotion, type Reaction } from "./usePetMotion";
import { findUpdate, UPDATE_NOTICE_KEY } from "../updater";
import "./sprites.css"; // Generated sprite indices.
import "./pet.css";

const YAWN_AFTER = 60_000;
// Silent reply timeout.
const DISMISS_AFTER = 20_000;
// Silent point timeout.
const POINT_DISMISS = 10_000;
// Yawn duration.
const YAWN_LENGTH = 3_200;
// Break settle delay.
const DOZE_AFTER = 1_500;
const PARTICLES = [0, 1, 2, 3, 4];
const MIC_BAR_REST = 0.19;
const MIC_BARS = [0.19, 0.34, 0.62, 1, 0.62, 0.34, 0.19];
// Alert timeout.
const ALERT_CAP = 120_000;
const UPDATE_CHECK_DELAY = 8_000;
const UPDATE_NOTICE_LENGTH = 15_000;

/** Meeting badge labels. */
const MEETING_LABEL: Record<string, string> = {
  starting: "Getting ready",
  recording: "Notes",
  paused: "Paused",
  finalizing: "Saving notes",
};

/** Open panel. */
type Panel = "pomodoro" | "reminders" | "meeting" | null;

/** Nap state. */
type Nap = "awake" | "yawn" | "sleeping";

/** Nap override. */
type Hold = "awake" | "sleep" | null;

type GuideAck = { accepted: boolean; arrived: boolean };

// Monotonic guide revision.
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
  // Reactions win.
  if (reaction === "angry") return "angry";
  if (reaction === "drag") return "listening";
  if (reaction === "pet") return "petting";
  if (reaction === "hunt") return "hunt";
  // Quiet precedes alerts.
  if (quiet) return "peek";
  // Alerts persist until dismissed.
  if (alerting) return "alert";
  // Looking shares thinking art.
  if (state === "looking") return "thinking";
  if (state !== "idle") return state;
  return nap === "awake" ? "idle" : nap;
}

export default function Pet() {
  useCoat();
  const {
    connected,
    state,
    microphone,
    micLevel,
    transcript,
    reply,
    error,
    writing,
    speak,
    reminder,
    point,
    // Socket timer request.
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
  // Show only blocked drafts.
  const writingPanel = writing && writing.status === "blocked" ? writing : null;
  // Local pomodoro alert.
  const [fired, setFired] = useState("");
  // Alerts queued while quiet.
  const [waiting, setWaiting] = useState<string[]>([]);
  const [availableUpdate, setAvailableUpdate] = useState<string | null>(null);
  const [updateNotice, setUpdateNotice] = useState("");
  const timer = usePomodoro(setFired);
  const alert = fired || reminder || updateNotice;
  useEffect(() => { if (meetingActive && alert) setPanel("meeting"); }, [meetingActive, alert]);
  // Activity overrides naps.
  const holdMode: Hold =
    meetingActive || panel !== null || writingPanel !== null || alert !== "" || state !== "idle"
      ? "awake"
      : !timer.running
        ? null
        : timer.phase === "focus"
          ? "awake"
          : "sleep";
  const hold = useRef(holdMode);
  const naps = useRef<number[]>([]);
  // Survive StrictMode remounts.
  const held = useRef(false);

  // Mellow inactivity timer.
  const wake = useCallback(() => {
    naps.current.forEach(clearTimeout);
    setNap("awake");
    // Follow nap state.
    const yawnAt = hold.current === "sleep" ? DOZE_AFTER : YAWN_AFTER;
    naps.current = [
      // Clear before sleep.
      setTimeout(() => {
        setNap("yawn");
        if (hold.current !== "awake") clear();
      }, yawnAt),
      setTimeout(() => setNap("sleeping"), yawnAt + YAWN_LENGTH),
    ];
  }, [clear]);

  // Wake after onboarding.
  useEffect(() => {
    const stop = listen("pet-wake", wake);
    return () => {
      stop.then((off) => off()).catch(() => {});
    };
  }, [wake]);

  // Sidecar activity wakes Mellow.
  useEffect(() => {
    wake();
  }, [state, wake]);

  useEffect(() => () => naps.current.forEach(clearTimeout), []);

  // Re-arm the nap timer.
  useEffect(() => {
    hold.current = holdMode;
    wake();
  }, [holdMode, wake]);

  // Visible nap pose.
  const pose: Nap = holdMode === "awake" ? "awake" : nap;

  const dismiss = useCallback(() => {
    setFired("");
    setUpdateNotice("");
    dismissReminder();
  }, [dismissReminder]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void findUpdate()
        .then(async (update) => {
          if (!update) return;
          const version = update.version;
          await update.close();
          if (!cancelled) setAvailableUpdate(version);
        })
        .catch(() => undefined);
    }, UPDATE_CHECK_DELAY);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  // Auto-dismiss alerts.
  useEffect(() => {
    if (!alert) return;
    const timeout = setTimeout(dismiss, ALERT_CAP);
    return () => clearTimeout(timeout);
  }, [alert, dismiss]);

  // Handle spoken timers.
  useEffect(() => {
    if (!asked) return;
    if (asked.action === "stop") {
      timer.stop();
      return;
    }
    timer.start(asked.minutes ?? undefined);
    setPanel("pomodoro");
  }, [asked]);

  // Open native-menu panels.
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
  // Quiet edge state.
  const { quiet, setQuiet, toggleQuiet } = motion;
  useEffect(() => {
    if (
      !availableUpdate ||
      updateNotice ||
      quiet ||
      meetingActive ||
      state !== "idle" ||
      fired ||
      reminder
    ) return;
    if (localStorage.getItem(UPDATE_NOTICE_KEY) === availableUpdate) {
      setAvailableUpdate(null);
      return;
    }
    localStorage.setItem(UPDATE_NOTICE_KEY, availableUpdate);
    setUpdateNotice(`Mellow ${availableUpdate} is ready. Open Settings → Updates.`);
  }, [availableUpdate, updateNotice, quiet, meetingActive, state, fired, reminder]);

  useEffect(() => {
    if (!updateNotice) return;
    const timeout = window.setTimeout(() => setUpdateNotice(""), UPDATE_NOTICE_LENGTH);
    return () => window.clearTimeout(timeout);
  }, [updateNotice]);
  // Release the overlay before Settings.
  const viewMeetings = useCallback(
    (id: string | null) => {
      setDismissedMeetingId(id);
      setPanel(null);
      motion.releaseOverlay();
      void viewMeeting(id);
    },
    [motion],
  );

  // Mirror tray releases.
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
  // Disable pointing while hidden.
  const pointing = point !== null && !quiet && pose === "awake";
  // Wait for bone arrival.
  const [landed, setLanded] = useState(false);
  // Track the pointing turn.
  const pointed = useRef(false);
  // Dismiss after playback.
  const spokeThisTurn = useRef(false);
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

  // Clear the guide on unmount.
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
        // Fall back to dialogue.
        if (revision === guideRevision.current) setLanded(true);
      });
  }, [point]);

  useEffect(() => {
    void invoke("guide_set_quiet", { quiet: quiet !== null }).catch((error) =>
      console.error("[mellow] guide visibility failed:", error),
    );
  }, [quiet]);

  // Track turn playback.
  useEffect(() => {
    if (state === "listening" || state === "thinking") {
      pointed.current = false;
      spokeThisTurn.current = false;
    } else if (state === "talking") {
      spokeThisTurn.current = true;
    }
  }, [state]);

  // Queue alerts while quiet.
  useEffect(() => {
    if (!quiet || (!fired && !reminder)) return;
    setWaiting((queue) => [...queue, fired, reminder].filter(Boolean));
    setFired("");
    dismissReminder();
  }, [quiet, fired, reminder, dismissReminder]);

  // Close panels when quiet.
  useEffect(() => {
    if (quiet) setPanel(null);
  }, [quiet]);

  useEffect(() => {
    if (quiet || alert || waiting.length === 0) return;
    setFired(waiting[0]);
    setWaiting((queue) => queue.slice(1));
  }, [quiet, alert, waiting]);

  // Handle the PTT hotkey.
  useEffect(() => {
    const stop = listen<boolean>("ptt", ({ payload: down }) => {
      if (meetingActive) return;
      // The sidecar keeps this press pending while the microphone wakes.
      if (!down && !held.current) return;
      // Ignore key-repeat.
      if (down === held.current) return;
      held.current = down;
      if (down) {
        // Exit quiet mode.
        setQuiet(null);
        wake();
        // Clear stale dialogue.
        clear();
        send({ type: "ptt_start" });
      } else {
        send({ type: "ptt_end" });
      }
    });
    return () => {
      stop.then((off) => off()).catch(() => {});
    };
  }, [send, wake, clear, setQuiet, meetingActive]);

  // Keep the mic ready while awake.
  const listening = !meetingActive && nap !== "sleeping" && !quiet;
  useEffect(() => {
    if (connected) send({ type: "awake", value: listening });
  }, [connected, listening, send]);

  // Open the native context menu.
  const openMenu = useCallback(
    (event: React.MouseEvent) => {
      // Suppress the WebView menu.
      event.preventDefault();
      wake();
      // Sync menu labels.
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

  // Start a new conversation.
  useEffect(() => {
    const stop = listen("new-chat", () => {
      clear();
      send({ type: "new_conversation" });
    });
    return () => {
      stop.then((off) => off()).catch(() => {});
    };
  }, [send, clear]);

  // Dialogue priority.
  const said = alert || error || reply || transcript;

  // Clear completed exchanges.
  useEffect(() => {
    // Idle follows playback.
    if (state !== "idle" || !said || alert) return;
    if (spokeThisTurn.current) {
      pointed.current = false;
      spokeThisTurn.current = false;
      clear();
      return;
    }
    const reading = setTimeout(
      () => {
        pointed.current = false;
        spokeThisTurn.current = false;
        clear();
      },
      pointing || pointed.current ? POINT_DISMISS : DISMISS_AFTER,
    );
    return () => clearTimeout(reading);
    // Points restart the timer.
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
      // Open dialogue before text.
      const payload: GuideDialogue | null = remoteDialogue && spot
        ? {
            text: said,
            error: error !== "",
            side: spot.side,
            lift: spot.lift,
          }
        : null;
      if (payload) {
        // Back up guide text.
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
      // Drop stale updates.
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
        {!meetingActive && connected && state === "listening" && !quiet && pose === "awake" && (
          <div className="mic-meter" role="status" aria-label="Listening">
            {MIC_BARS.map((weight, index) => (
              <i
                key={index}
                aria-hidden="true"
                style={{
                  transform: `scaleY(${MIC_BAR_REST + micLevel * (weight - MIC_BAR_REST)})`,
                }}
              />
            ))}
          </div>
        )}
        {/* Waiting marker while quiet (not a count). */}
        {quiet && waiting.length > 0 && <i className="quiet-dot" />}
        {!panel && !writingPanel && (meetingActive || meetingSaved) && (
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
        {writingPanel && !meetingActive && !quiet && (
          <div className="panel-anchor" ref={motion.panelRef}>
            <WritingPanel draft={writingPanel} send={send} />
          </div>
        )}
        {panel && !writingPanel && (
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
        {!panel && !writingPanel && !quiet && !pointing && pose === "awake" &&
          // Hide empty dialogue.
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
