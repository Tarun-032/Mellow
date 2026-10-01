import { useCallback, useEffect, useRef, useState } from "react";
import { emit, listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import type { GuideOutcome } from "./guidePresentation";

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
  | { type: "state"; state: PetState }
  | { type: "microphone"; state: MicrophoneState }
  | { type: "mic_level"; level: number }
  | { type: "transcript"; text: string }
  | { type: "reply_chunk"; text: string }
  | { type: "speak"; value: boolean }
  | { type: "remind"; text: string; id: string }
  | { type: "pong"; echo: string }
  | { type: "capture"; phase: "begin" | "end" }
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
      if (!enabled) retirePoint();
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
  }, [connected, retirePoint]);

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
            // Hide before capture.
            if (msg.phase === "begin") {
              // Keep the turn monitor.
              void emit("pet-capture", { hidden: true }).then(() => {
                sock.send(JSON.stringify({ type: "capture_ready" }));
              }).catch((error) => {
                console.error("[mellow] could not prepare screen capture", error);
                sock.send(JSON.stringify({ type: "capture_ready" }));
              });
            } else {
              void emit("pet-capture", { hidden: false });
            }
            break;
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
        if (!disposed) timer = setTimeout(connect, 1000);
      };
    };

    connect();
    return () => {
      disposed = true;
      clearTimeout(timer);
      ws.current?.close();
    };
  }, []);

  /** Clear dialogue. */
  const clear = useCallback(() => {
    setTranscript("");
    setReply("");
    setError("");
    setReminder("");
    retirePoint();
  }, [retirePoint]);

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
    dismissReminder,
  };
}
