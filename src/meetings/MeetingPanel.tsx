import { useEffect, useState } from "react";
import { emit } from "@tauri-apps/api/event";
import { request } from "../ui/fields";
import { clock, type MeetingStatus } from "./useMeeting";
import "./meeting-panel.css";

type Device = { id: number; name: string; default: boolean };
type Devices = { inputs: Device[]; outputs: Device[] };

export function MeetingPanel({ status, connectionError, refresh, onClose, onView, alert, onDismissAlert }: {
  status: MeetingStatus | null; connectionError: string;
  refresh: () => Promise<MeetingStatus>; onClose: () => void;
  onView: (id: string | null) => void;
  alert: string; onDismissAlert: () => void;
}) {
  const [title, setTitle] = useState("");
  const [devices, setDevices] = useState<Devices | null>(null);
  const [microphone, setMicrophone] = useState("");
  const [output, setOutput] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  // Speech settings still have to load before Start: a failure here means we
  // cannot tell how audio would be transcribed, so the button stays disabled.
  const [ready, setReady] = useState(false);
  const [levels, setLevels] = useState<Record<string, number> | null>(null);
  const active = status?.active;
  useEffect(() => {
    request<Devices>("/meetings/devices").then(setDevices).catch(e => setError(e.message));
    request<{ settings: { stt: { mode: string; provider: string; input_device: string | null } } }>("/config")
      .then(() => setReady(true))
      .catch(e => setError(e.message));
  }, []);
  // Focusable up front: a <select> opens its native popup before onFocus lands,
  // and the set_focus that followed was dismissing the open list.
  useEffect(() => {
    void emit("pet-focus", { focus: true });
    return () => { void emit("pet-focus", { focus: false }); };
  }, []);

  const action = async (name: string) => {
    setBusy(true); setError("");
    try {
      const result = await request<{ levels?: Record<string, number> }>(`/meetings/${name}`, {
        method: "POST", body: JSON.stringify(name === "start" || name === "levels" ? {
          title, microphone: microphone === "" ? null : Number(microphone), output: output === "" ? null : Number(output),
        } : {}),
      });
      if (name === "levels" && result.levels) setLevels(result.levels);
      await refresh();
      // Recording and saved notes use the compact badge.
      if (name === "start" || name === "stop") onClose();
    } catch (e) { setError(String(e instanceof Error ? e.message : e)); }
    finally { setBusy(false); }
  };

  return <section className="panel meeting-panel" aria-label="Meeting transcription">
    <div className="panel__bar">
      <span className="panel__title">{active ? "Meeting in progress" : "Transcribe a meeting"}</span>
      <button type="button" className="panel__btn panel__btn--ghost" onClick={onClose} aria-label="Close meeting controls">×</button>
    </div>
    {alert && <div role="status"><p>{alert}</p><button type="button" className="panel__btn" onClick={onDismissAlert}>Dismiss reminder</button></div>}
    {active ? <>
      <div className="meeting-panel__status" role="status">
        <span>{status.status === "recording" ? "Recording" : status.status === "paused" ? "Paused" : status.status === "finalizing" ? "Finishing transcript…" : "Preparing audio…"}</span>
        <time>{clock(status.duration)}</time>
      </div>
      {status.status === "recording" && <div className="meeting-panel__levels">
        {["You", "Other participants"].map(name => <label key={name}>{name}
          <meter min={0} max={1} value={Math.min(1, (status.levels[name] || 0) * 4)} aria-label={`${name} audio level`} />
        </label>)}
      </div>}
      <p className="panel__hint">{status.pending ? `${status.pending} audio chunks waiting to finish.` : "Transcript saves automatically. Audio is not kept."}</p>
      <div className="panel__row panel__row--buttons">
        <button type="button" className="panel__btn" disabled={busy || status.status === "finalizing" || status.status === "starting"}
          onClick={() => void action(status.status === "paused" ? "resume" : "pause")}>{status.status === "paused" ? "Resume" : "Pause"}</button>
        <button type="button" className="panel__btn" disabled={busy || status.status === "finalizing" || status.status === "starting"}
          onClick={() => void action("stop")}>Stop & save</button>
        <button type="button" className="panel__btn panel__btn--ghost" onClick={() => onView(status.id)}>View</button>
      </div>
    </> : <>
      <label>Title <span className="panel__hint">(optional)</span>
        <input className="panel__input" value={title} onChange={e => setTitle(e.target.value)} maxLength={160} placeholder="Weekly catch-up" />
      </label>
      <label>Microphone
        <select className="panel__input" value={microphone} onChange={e => setMicrophone(e.target.value)}>
          <option value="">Configured microphone / Windows default</option>
          {devices?.inputs.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
      </label>
      <label>Meeting sound
        <select className="panel__input" value={output} onChange={e => setOutput(e.target.value)}>
          <option value="">Windows default output</option>
          {devices?.outputs.map(d => <option key={d.id} value={d.id}>{d.name.replace(" [Loopback]", "")}</option>)}
        </select>
      </label>
      <div className="panel__row panel__row--buttons">
        <button type="button" className="panel__btn panel__btn--wide" disabled={busy || !devices || !!connectionError} onClick={() => void action("levels")}>Check audio levels (2s)</button>
      </div>
      {levels && <p className="panel__hint">Microphone: {levels.You > 0.01 ? "sound detected" : "quiet"}. Meeting sound: {levels["Other participants"] > 0.01 ? "sound detected" : "quiet"}. Play meeting audio and speak while checking.</p>}
      <div className="panel__row panel__row--buttons">
        <button type="button" className="panel__btn panel__btn--wide" disabled={busy || !devices || !ready || !!connectionError}
          onClick={() => void action("start")}>{busy ? "Preparing audio…" : "Start transcription"}</button>
      </div>
    </>}
    {(error || connectionError || status?.warning) && <p className="panel__hint panel__hint--error" role="alert">{error || connectionError || status?.warning}</p>}
  </section>;
}
