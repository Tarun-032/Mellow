import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import type { GuideOutcome } from "./guidePresentation";
import { nextAnnotationRevision, type AnnotationScene } from "./annotationScene";

const URL = "ws://127.0.0.1:8765/ws";

export type PetState = "idle" | "listening" | "thinking" | "looking" | "talking";
export type MicrophoneState = "warming" | "ready" | "off";

type Monitor = { left: number; top: number; width: number; height: number };

export type WritingStatus = {
  type: "writing";
  id: string;
// Insertion was not verified.
  status: "idle" | "thinking" | "inserting" | "inserted" | "sent" | "blocked" | "uncertain";
  text: string;
  message: string;
  retry: boolean;
};

type Incoming =
  | WritingStatus
  | { type: "guide"; waiting: boolean }
  | { type: "state"; state: PetState; interrupted?: boolean }
  | { type: "microphone"; state: MicrophoneState }
  | { type: "mic_level"; level: number }
  | { type: "transcript"; text: string }
  | { type: "reply_chunk"; text: string }
  | { type: "speak"; value: boolean }
  | { type: "remind"; text: string; id: string }
  | { type: "pong"; echo: string }
  | { type: "capture"; phase: "begin" | "end"; capture_id: string }
  | { type: "drawing"; scene: AnnotationScene | null; preserve_pen?: boolean }
  | { type: "drawing_prepare"; monitor: { left: number; top: number } | null }
  | { type: "drawing_finish" }
  | { type: "drawing_keepalive"; presentation_id: string }
  | { type: "point"; nx: number | null; ny?: number; label?: string; monitor?: Monitor; presentation_id?: string }
  | { type: "pomodoro"; action: "start" | "stop"; minutes?: number | null }
  | { type: "error"; message: string };

/** Normalized bone target. */
export type Point = {
  nx: number; ny: number; label: string; monitor?: Monitor;
  acknowledge: (outcome: GuideOutcome) => void;
};

/** Sidecar connection. */
export function useSocket() {
  const [connected, setConnected] = useState(false);
  const [aiEnabled, setAiEnabled] = useState(false);
  const aiEnabledRef = useRef(false);
  const drawingAllowed = useRef(false);
  const drawing = useRef<{ revision: number; id: string; socket: WebSocket; finished: boolean;
    readyPending?: boolean; readyForwarded?: boolean; completePending?: boolean } | null>(null);
  const drawingRevision = useRef(0);
  const [drawingPen, setDrawingPen] = useState(false);
  const clearDrawing = useCallback((keepPen = false, reason = "cleared") => {
    const previous = drawing.current;
    drawing.current = null;
    if (previous && previous.socket.readyState === WebSocket.OPEN) {
      previous.socket.send(JSON.stringify({ type: "drawing_ack", presentation_id: previous.id, outcome: "failed", reason }));
    }
    if (!keepPen) setDrawingPen(false);
    drawingRevision.current = nextAnnotationRevision();
    void invoke("annotation_clear", { revision: drawingRevision.current, keepPen })
      .catch((error) => console.error("[mellow] could not clear drawings", error));
  }, []);
  const setDrawingAllowed = useCallback((allowed: boolean) => {
    drawingAllowed.current = allowed;
    if (!allowed) clearDrawing();
  }, [clearDrawing]);
  const [state, setState] = useState<PetState>("idle");
  // Microphone readiness.
  const [microphone, setMicrophone] = useState<MicrophoneState>("warming");
  const [micLevel, setMicLevel] = useState(0);
  const [transcript, setTranscript] = useState("");
  const [reply, setReply] = useState("");
  // Current error.
  const [error, setError] = useState("");
  const [writing, setWriting] = useState<WritingStatus | null>(null);
  // Sidecar voice flag.
  const [speak, setSpeak] = useState(true);
  // Fired reminder.
  const [reminder, setReminder] = useState("");
  // Current point target.
  const [point, setPoint] = useState<Point | null>(null);
  const activePoint = useRef<Point | null>(null);
  const retirePoint = useCallback(() => {
    activePoint.current?.acknowledge("failed");
    activePoint.current = null;
    setPoint(null);
  }, []);
  const [guideWaiting, setGuideWaiting] = useState(false);
  // Spoken pomodoro request.
  const [timer, setTimer] = useState<{ action: "start" | "stop"; minutes: number | null; n: number } | null>(null);
  const ws = useRef<WebSocket | null>(null);

  // Follow saved settings only; an unsaved engine selection must not hide the guide.
  useEffect(() => {
    let alive = true;
    let revision = 0;
    const update = (enabled: boolean) => {
      if (!alive) return;
      aiEnabledRef.current = enabled;
      setAiEnabled(enabled);
      if (!enabled) { retirePoint(); clearDrawing(); }
    };
    const stop = listen<boolean>("ai-enabled", ({ payload }) => {
      if (typeof payload !== "boolean") return;
      revision += 1;
      update(payload);
    });
    // Subscribe before reading so an older response cannot undo a newer save.
    void stop.then(async () => {
      if (!connected || !alive) return;
      const current = revision;
      const response = await fetch("http://127.0.0.1:8765/config");
      if (!response.ok) return;
      const body = await response.json();
      if (revision === current && typeof body?.settings?.ai_enabled === "boolean") {
        update(body.settings.ai_enabled);
      }
    }).catch((error) => console.error("[mellow] could not read guide visibility:", error));
    return () => {
      alive = false;
      void stop.then((off) => off()).catch(() => {});
    };
  }, [connected, retirePoint, clearDrawing]);

  useEffect(() => {
    const stop = listen<{ revision: number; presentation_id: string; outcome: string; reason?: string; scale?: number; remaining_ms?: number; occlusions?: number[][] }>("drawing-receipt", ({ payload }) => {
      const active = drawing.current;
      if (!active || active.revision !== payload.revision || active.id !== payload.presentation_id
        || ws.current !== active.socket || active.socket.readyState !== WebSocket.OPEN) return;
      if (payload.outcome === "position") {
        active.socket.send(JSON.stringify({ type: "drawing_ack", presentation_id: active.id, outcome: "position", occlusions: payload.occlusions }));
        return;
      }
      if (!["ready", "complete", "failed"].includes(payload.outcome)) return;
      if (payload.outcome === "complete" && !active.readyForwarded) {
        // Measuring the pet's bounds is asynchronous. A short/reduced-motion
        // drawing can finish first; forward its completion only after ready.
        if (active.readyPending) active.completePending = true;
        return;
      }
      if (payload.outcome === "ready") {
        if (active.readyPending || active.readyForwarded) return;
        active.readyPending = true;
        const rects: number[][] = [];
        const body = document.querySelector(".pet-body")?.getBoundingClientRect();
        if (body) rects.push([body.x - 12, body.y - 12, body.width + 24, body.height + 24]);
        const bubble = document.querySelector(".pet-root > .bubble")?.getBoundingClientRect();
        const root = document.querySelector(".pet-root")?.getBoundingClientRect();
        // The reply arrives after readiness; reserve the existing CSS maximum
        // bubble width/reading height rather than just its initial dots.
        // A reply can create the bubble after this receipt. Pet.css anchors it
        // 91px from the root's right and 34px below its top, even when absent.
        const right = bubble?.right ?? (root ? root.right - 91 : null);
        const bottom = bubble?.bottom ?? (root ? root.top + 34 : null);
        if (right !== null && bottom !== null) rects.push([right - 432, bottom - 232, 444, 244]);
        void invoke<number[][]>("annotation_occlusion", { rects }).then((occlusions) => {
          if (drawing.current !== active || ws.current !== active.socket || active.socket.readyState !== WebSocket.OPEN) return;
          active.socket.send(JSON.stringify({ type: "drawing_ack", presentation_id: active.id, outcome: "ready", occlusions, scale: payload.scale, remaining_ms: payload.remaining_ms }));
          active.readyForwarded = true;
          if (active.completePending) {
            active.socket.send(JSON.stringify({ type: "drawing_ack", presentation_id: active.id, outcome: "complete" }));
          }
        }).catch((error) => {
          console.error("[mellow] could not measure pet bounds", error);
          if (drawing.current === active) clearDrawing(false, "presentation_failed");
        });
        return;
      }
      const reason = payload.outcome === "failed"
        ? (["expired", "source_changed", "pet_hidden", "display_changed", "replaced", "cleared"].includes(payload.reason ?? "") ? payload.reason : "native_failure")
        : undefined;
      active.socket.send(JSON.stringify({ type: "drawing_ack", presentation_id: active.id, outcome: payload.outcome, reason }));
      if (payload.outcome === "failed") { drawing.current = null; setDrawingPen(false); }
    });
    return () => { void stop.then((off) => off()).catch(() => {}); };
  }, [clearDrawing]);

  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;

    const connect = () => {
      const sock = new WebSocket(URL);
      ws.current = sock;

      sock.onopen = () => {
        setConnected(true);
        setMicrophone("warming");
      };

      sock.onmessage = async (e) => {
        if (disposed || ws.current !== sock) return;
        const msg: Incoming = JSON.parse(e.data);
        switch (msg.type) {
          case "guide":
            setGuideWaiting(msg.waiting);
            break;
          case "writing":
            setWriting(msg.status === "idle" ? null : msg);
            break;
          case "state":
            setState(msg.state);
            if (msg.state === "idle" && msg.interrupted) {
              // A stopped visual explanation has no final playback/reading
              // event. Clear its old dialogue even if audio never began.
              setTranscript("");
              setReply("");
              setError("");
              setGuideWaiting(false);
              retirePoint();
              clearDrawing();
            }
            if (msg.state !== "listening") setMicLevel(0);
            break;
          case "microphone":
            setMicrophone(msg.state);
            break;
          case "mic_level":
            setMicLevel(Math.max(0, Math.min(1, msg.level)));
            break;
          case "transcript":
            setTranscript(msg.text);
            setReply("");
            break;
          case "reply_chunk":
            setReply((r) => r + msg.text);
            break;
          case "speak":
            setSpeak(msg.value);
            break;
          case "remind":
            setReminder(msg.text);
            break;
          case "pong":
            console.log("[mellow] pong:", msg.echo);
            break;
          case "capture":
            void invoke<boolean>("capture_prepare", { revision: nextAnnotationRevision(),
              captureId: msg.capture_id, hidden: msg.phase === "begin" }).then((ok) => {
              if (msg.phase === "begin" && !disposed && ws.current === sock && sock.readyState === WebSocket.OPEN) {
                sock.send(JSON.stringify({ type: "capture_ready", capture_id: msg.capture_id, ok }));
              }
            }).catch((error) => {
              console.error("[mellow] could not prepare screen capture", error);
              if (msg.phase === "begin" && !disposed && ws.current === sock && sock.readyState === WebSocket.OPEN) {
                sock.send(JSON.stringify({ type: "capture_ready", capture_id: msg.capture_id, ok: false }));
              }
            });
            break;
          case "drawing_prepare":
            // Build the monitor's overlay while the answer is planned, so the
            // first drawing does not wait for WebView2 to start.
            if (aiEnabledRef.current && drawingAllowed.current && msg.monitor) {
              void invoke("annotation_prepare", { left: msg.monitor.left, top: msg.monitor.top })
                .catch((error) => console.error("[mellow] could not prepare drawing overlay", error));
            }
            break;
          case "drawing": {
            clearDrawing(Boolean(msg.scene || msg.preserve_pen));
            if (!msg.scene) break;
            const active = { revision: nextAnnotationRevision(), id: msg.scene.presentation_id, socket: sock, finished: false };
            drawing.current = active;
            drawingRevision.current = active.revision;
            if (!aiEnabledRef.current || !drawingAllowed.current) { clearDrawing(); break; }
            setDrawingPen(true);
            void invoke<boolean>("annotation_present", { revision: active.revision, scene: msg.scene }).then((accepted) => {
              if (!accepted && drawing.current === active) clearDrawing(false, "presentation_failed");
            }).catch((error) => {
              console.error("[mellow] could not present drawing", error);
              if (drawing.current === active) clearDrawing(false, "presentation_failed");
            });
            break;
          }
          case "drawing_finish": {
            const active = drawing.current;
            if (active) active.finished = true;
            if (drawingRevision.current) void invoke("annotation_finish", { revision: active?.revision ?? drawingRevision.current }).catch(() => clearDrawing());
            setDrawingPen(false);
            break;
          }
          case "drawing_keepalive": {
            const active = drawing.current;
            if (!active || active.finished || active.id !== msg.presentation_id || active.socket !== sock) break;
            // Preserve completed strokes during further narration; an old
            // receipt cannot renew a replacement scene or reborrow its bone.
            void invoke<boolean>("annotation_renew", { revision: active.revision }).then((accepted) => {
              if (!accepted && drawing.current === active && !active.finished) clearDrawing(false, "presentation_failed");
            }).catch((error) => {
              console.error("[mellow] could not extend active drawing", error);
              if (drawing.current === active && !active.finished) clearDrawing(false, "presentation_failed");
            });
            break;
          }
          case "pomodoro":
            setTimer((current) => ({
              action: msg.action,
              minutes: msg.minutes ?? null,
              n: (current?.n ?? 0) + 1,
            }));
            break;
          case "point":
            // Bind receipts to this socket and object, never a replacement connection.
            activePoint.current = null;
            if (msg.nx === null || !aiEnabledRef.current) {
              if (msg.nx !== null && msg.presentation_id && sock.readyState === WebSocket.OPEN) {
                sock.send(JSON.stringify({ type: "guide_ack", presentation_id: msg.presentation_id, outcome: "failed" }));
              }
              setPoint(null);
            } else {
              let acknowledged = false;
              const target: Point = {
                nx: msg.nx,
                ny: msg.ny ?? 0,
                label: msg.label ?? "",
                monitor: msg.monitor,
                acknowledge: (outcome) => {
                  if (acknowledged || activePoint.current !== target || ws.current !== sock || sock.readyState !== WebSocket.OPEN || !msg.presentation_id) return;
                  acknowledged = true;
                  sock.send(JSON.stringify({ type: "guide_ack", presentation_id: msg.presentation_id, outcome }));
                },
              };
              activePoint.current = target;
              setPoint(target);
            }
            break;
          case "error":
            // Log visible errors.
            console.error("[mellow]", msg.message);
            setError(msg.message);
            break;
        }
      };

      sock.onerror = () => sock.close();
      sock.onclose = () => {
        if (disposed || ws.current !== sock) return;
        setConnected(false);
        setWriting(null);
        setMicrophone("off");
        setMicLevel(0);
        // Clear disconnected guides.
        activePoint.current = null;
        setPoint(null);
        setGuideWaiting(false);
        clearDrawing();
        if (!disposed) timer = setTimeout(connect, 1000);
      };
    };

    connect();
    return () => {
      disposed = true;
      clearTimeout(timer);
      ws.current?.close();
      clearDrawing();
    };
  }, [clearDrawing]);

  /** Dismiss finished dialogue without retiring useful on-screen drawings. */
  const dismissDialogue = useCallback(() => {
    setTranscript("");
    setReply("");
    setError("");
    setReminder("");
    retirePoint();
  }, [retirePoint]);

  /** Explicit reset also retires drawings (new turn, sleep, hide, etc.). */
  const clear = useCallback(() => {
    dismissDialogue();
    clearDrawing();
  }, [dismissDialogue, clearDrawing]);

  /** Dismiss a reminder. */
  const dismissReminder = useCallback(() => setReminder(""), []);

  // Stable sender.
  const send = useCallback((msg: object) => {
    const transmit = (payload: object) => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify(payload));
      }
    };
    const type = (msg as { type?: string }).type;
    if (type !== "ptt_end" && type !== "text") {
      transmit(msg);
      return;
    }

    // Lock the turn monitor.
    void invoke<Monitor>("cursor_monitor")
      .then((monitor) => transmit({ ...msg, monitor }))
      .catch((error) => {
        console.error("[mellow] could not lock cursor monitor", error);
        transmit(msg); // Use the sidecar fallback.
      });
  }, []);

  return {
    connected,
    aiEnabled,
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
    guideWaiting,
    timer,
    send,
    clear,
    dismissDialogue,
    dismissReminder,
    setDrawingAllowed,
    drawingPen,
  };
}
