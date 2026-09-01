import { useCallback, useEffect, useRef, useState } from "react";
import { emit } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";

const URL = "ws://127.0.0.1:8765/ws";

export type PetState = "idle" | "listening" | "thinking" | "looking" | "talking";
export type MicrophoneState = "warming" | "ready" | "off";

type Monitor = { left: number; top: number; width: number; height: number };

type Incoming =
  | { type: "state"; state: PetState }
  | { type: "microphone"; state: MicrophoneState }
  | { type: "transcript"; text: string }
  | { type: "reply_chunk"; text: string }
  | { type: "speak"; value: boolean }
  | { type: "remind"; text: string; id: string }
  | { type: "pong"; echo: string }
  | { type: "capture"; phase: "begin" | "end" }
  | { type: "point"; nx: number | null; ny?: number; label?: string; monitor?: Monitor }
  | { type: "pomodoro"; action: "start" | "stop"; minutes?: number | null }
  | { type: "error"; message: string };

/** Bone target as a fraction of the captured monitor. */
export type Point = { nx: number; ny: number; label: string; monitor?: Monitor };

/** Talks to the Python sidecar. Retries forever so start order doesn't matter. */
export function useSocket() {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<PetState>("idle");
  // Mic may still be warming after the socket connects.
  const [microphone, setMicrophone] = useState<MicrophoneState>("warming");
  const [transcript, setTranscript] = useState("");
  const [reply, setReply] = useState("");
  // Errors are their own state (not folded into reply).
  const [error, setError] = useState("");
  // speak mirrors the sidecar flag.
  const [speak, setSpeak] = useState(true);
  // Fired reminder from the sidecar clock.
  const [reminder, setReminder] = useState("");
  // Current point target, or null.
  const [point, setPoint] = useState<Point | null>(null);
  // Spoken pomodoro requests land here (not in Python).
  const [timer, setTimer] = useState<{ action: "start" | "stop"; minutes: number | null; n: number } | null>(null);
  const ws = useRef<WebSocket | null>(null);

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
        const msg: Incoming = JSON.parse(e.data);
        switch (msg.type) {
          case "state":
            setState(msg.state);
            break;
          case "microphone":
            setMicrophone(msg.state);
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
            // Hide before screenshot so Mellow isn't in the capture.
            if (msg.phase === "begin") {
              // Turn already owns the cursor's monitor.
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
            setPoint(
              msg.nx === null
                ? null
                : {
                    nx: msg.nx,
                    ny: msg.ny ?? 0,
                    label: msg.label ?? "",
                    monitor: msg.monitor,
                  },
            );
            break;
          case "error":
            // Devtools trail; sentence also goes on screen.
            console.error("[mellow]", msg.message);
            setError(msg.message);
            break;
        }
      };

      sock.onerror = () => sock.close();
      sock.onclose = () => {
        setConnected(false);
        setMicrophone("off");
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

  /** Clear on-screen dialogue (e.g. when nodding off). */
  const clear = useCallback(() => {
    setTranscript("");
    setReply("");
    setError("");
    setReminder("");
    setPoint(null);
  }, []);

  /** Acknowledge a reminder without wiping the conversation underneath it. */
  const dismissReminder = useCallback(() => setReminder(""), []);

  // Stable send identity for effect deps.
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

    // Snapshot on release; don't wait on model latency.
    void invoke<Monitor>("cursor_monitor")
      .then((monitor) => transmit({ ...msg, monitor }))
      .catch((error) => {
        console.error("[mellow] could not lock cursor monitor", error);
        transmit(msg); // sidecar retains its measured active-monitor fallback
      });
  }, []);

  return {
    connected,
    state,
    microphone,
    transcript,
    reply,
    error,
    speak,
    reminder,
    point,
    timer,
    send,
    clear,
    dismissReminder,
  };
}
