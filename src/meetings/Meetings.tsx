import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { request } from "../ui/fields";
import { clock, MEETING_OPEN, MEETING_SELECTION, useMeeting } from "./useMeeting";
import "./meetings.css";

type Summary = { id: string; title: string; created: string; duration: number; status: string; warning: string; notes_status: string };
type Segment = { id: number; start: number; end: number; speaker: string; text: string };
type Detail = Summary & {
  segments: Segment[];
  notes: string; notes_status: string; notes_error: string; notes_progress: string; engine: string;
};
const live = (status: string) => ["starting", "recording", "paused", "finalizing"].includes(status);
const speakerLabel = (speaker: string) => speaker === "Other participants" ? "Other participant" : speaker;

function conversationTurns(segments: Segment[]): Segment[] {
  const turns: Segment[] = [];
  for (const segment of [...segments].sort((a, b) => a.start - b.start || a.id - b.id)) {
    const previous = turns[turns.length - 1];
    if (previous?.speaker === segment.speaker) {
      previous.text += ` ${segment.text}`;
      previous.end = Math.max(previous.end, segment.end);
    } else {
      turns.push({ ...segment });
    }
  }
  return turns;
}

export default function Meetings() {
  const [items, setItems] = useState<Summary[]>([]);
  // The list is the landing page. Only a deep link from Mellow, which sets
  // MEETING_OPEN alongside the id, opens straight onto one meeting.
  const [selected, setSelected] = useState<string | null>(
    () => localStorage.getItem(MEETING_OPEN) ? localStorage.getItem(MEETING_SELECTION) : null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [tab, setTab] = useState<"transcript" | "notes">("transcript");
  const [query, setQuery] = useState("");
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [notesConsent, setNotesConsent] = useState(false);
  const [destination, setDestination] = useState("");
  const meeting = useMeeting();
  const current = useRef(selected);
  current.current = selected;
  const refresh = useCallback(async () => {
    const list = await request<{ meetings: Summary[] }>("/meetings");
    setItems(list.meetings); setLoaded(true);
    const id = current.current;
    if (id) {
      const next = await request<Detail>(`/meetings/${id}`);
      if (current.current === id) setDetail(next);
    }
  }, []);

  useEffect(() => {
    let alive = true;
    let timer = 0;
    const poll = async () => {
      try { await refresh(); } catch (e) { if (alive) setError(e instanceof Error ? e.message : String(e)); }
      if (alive) timer = window.setTimeout(poll, 2000);
    };
    void poll();
    const stop = listen("open-meetings", () => setSelected(localStorage.getItem(MEETING_SELECTION)));
    request<{ settings: { ai_enabled: boolean; llm: { mode: string; provider: string; model: string } } }>("/config").then(({ settings }) => {
      const llm = settings.llm;
      setDestination(!settings.ai_enabled ? "Choose an answer engine in Settings first."
        : llm.mode === "local" ? `Your transcript will be processed locally by ${llm.model}.`
        : `Your transcript will be sent to ${llm.provider} (${llm.model || "agent default"}) to generate notes.`);
    }).catch(e => setError(e.message));
    return () => { alive = false; clearTimeout(timer); stop.then(off => off()).catch(() => {}); };
  }, [refresh]);

  useEffect(() => {
    setDetail(null); setConfirmDelete(false); setNotesConsent(false); setError(""); setNotice("");
    if (!selected) return;
    localStorage.setItem(MEETING_SELECTION, selected);
    let alive = true;
    request<Detail>(`/meetings/${selected}`).then(value => {
      if (alive) { setDetail(value); setTitle(value.title); }
    }).catch(e => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [selected]);

  const act = async (action: () => Promise<unknown>, message = "") => {
    setBusy(true); setError(""); setNotice("");
    try { await action(); await refresh(); setNotice(message); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  const exportFile = async (format: string) => {
    const result = await request<{ text: string; filename: string }>(`/meetings/${selected}/export?format=${format}`);
    const url = URL.createObjectURL(new Blob([result.text], { type: format === "json" ? "application/json" : "text/plain;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url; link.download = result.filename;
    document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const filtered = items.filter(item => item.title.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
  const turns = conversationTurns(detail?.segments ?? []);

  const banner = <>
    {meeting.status?.active && <div className="meetings-live" role="status">
      <span>Meeting {meeting.status.status} · {clock(meeting.status.duration)}</span>
      <button type="button" onClick={() => setSelected(meeting.status!.id)}>View active meeting</button>
      <button type="button" disabled={busy || meeting.status.status === "finalizing" || meeting.status.status === "starting"}
        onClick={() => void act(() => request("/meetings/stop", { method: "POST", body: "{}" }))}>Stop &amp; save</button>
    </div>}
    {(error || meeting.error) && <p className="meetings-error" role="alert">{error || meeting.error}</p>}
    {notice && <p role="status">{notice}</p>}
  </>;

  if (!selected) return <section className="meetings">
    <p className="settings-note">Right-click Mellow and choose Transcribe meeting. Your transcript and notes stay here until you delete them, independently of saved conversations.</p>
    {banner}
    <label className="meetings-find">Find a meeting<input type="search" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search titles" /></label>
    {!loaded ? <p role="status">Loading meetings…</p> : !items.length ? <p>No meetings yet. Start your first from Mellow’s right-click menu.</p> : !filtered.length ? <p>No matching meetings.</p> :
      <ul className="meetings-list">{filtered.map(item => <li key={item.id}>
        <button type="button" onClick={() => setSelected(item.id)}>
          <span className="meetings-row__title">{item.title}</span>
          <span className="meetings-row__when">{new Date(item.created).toLocaleString()}</span>
          <span className="meetings-row__stats">{clock(item.duration)} · {item.status}{item.notes_status === "ready" ? " · Notes ready" : ""}</span>
          <span className="meetings-row__go" aria-hidden="true">›</span>
        </button>
      </li>)}</ul>}
  </section>;

  return <section className="meetings">
    <button type="button" className="meetings-back" onClick={() => setSelected(null)}>← All meetings</button>
    {banner}
    <article className="meeting-detail" aria-label="Selected meeting">
      {!detail ? <p role="status">Loading transcript…</p> : <>
        <div className="meeting-title">
          <label>Meeting title<input value={title} onChange={e => setTitle(e.target.value)} maxLength={160} onKeyDown={e => { if (e.key === "Enter") e.preventDefault(); }} /></label>
          <button type="button" disabled={busy || !title.trim() || title === detail.title}
            onClick={() => void act(() => request(`/meetings/${selected}`, { method: "PUT", body: JSON.stringify({ title }) }), "Title saved.")}>Rename</button>
        </div>
        <p className="meeting-meta">{new Date(detail.created).toLocaleString()} · {clock(detail.duration)} · {detail.status}</p>
        {detail.warning && <p className="meetings-warning">{detail.warning}</p>}
        <div className="meeting-tabs" role="group" aria-label="Meeting view">
          <button type="button" aria-pressed={tab === "transcript"} onClick={() => setTab("transcript")}>Transcript</button>
          <button type="button" aria-pressed={tab === "notes"} onClick={() => setTab("notes")}>Notes</button>
        </div>
        {tab === "transcript" ? <>
          <p className="meeting-meta">Labels identify your microphone and the meeting audio—not individual speakers.</p>
          <div className="meeting-transcript">
            {!turns.length ? <p>{live(detail.status) ? "Listening. Your conversation appears here as speech is processed." : "No speech was transcribed. Check your microphone, meeting output and speech-to-text settings before recording again."}</p> :
              turns.map(segment => <section key={segment.id}>
                <div><strong>{speakerLabel(segment.speaker)}</strong></div><p>{segment.text}</p>
              </section>)}
          </div>
        </> : <>
          <p className="meeting-meta">{destination}</p>
          <label className="meeting-consent"><input type="checkbox" checked={notesConsent} onChange={e => setNotesConsent(e.target.checked)} /> Use my selected answer engine for these notes.</label>
          <button type="button" disabled={busy || live(detail.status) || detail.notes_status === "generating" || !detail.segments.length || !notesConsent || !destination}
            onClick={() => void act(() => request(`/meetings/${selected}/notes`, { method: "POST", body: "{}" }))}>
            {detail.notes_status === "generating" ? "Generating notes…" : detail.notes ? "Regenerate notes" : "Generate notes"}
          </button>
          {live(detail.status) && <p className="meeting-meta">Stop recording and let the remaining audio finish first.</p>}
          {detail.notes_status === "generating" && <p role="status">{detail.notes_progress || "Reading the transcript…"} You can leave this page.</p>}
          {detail.notes_error && <p className="meetings-error" role="alert">{detail.notes_error}</p>}
          {detail.notes ? <><div className="meeting-notes">{detail.notes.split("\n").filter(line => line.trim()).map((line, i) =>
            /^#{1,6}\s/.test(line) ? <h3 key={i}>{line.replace(/^#{1,6}\s*/, "")}</h3> : <p key={i}>{line}</p>)}</div><p className="meeting-meta">Generated with {detail.engine}. Review these notes against the transcript.</p></> : <p>Your summary, decisions and action items will appear here.</p>}
        </>}
        <div className="meeting-actions">
          <button type="button" disabled={busy} onClick={() => void act(() => navigator.clipboard.writeText(tab === "notes" ? detail.notes : turns.map(s => `${speakerLabel(s.speaker)}:\n${s.text}`).join("\n\n")), "Copied to clipboard.")}>Copy {tab}</button>
          {["md", "txt", "json"].map(format => <button type="button" key={format} disabled={busy} onClick={() => void act(() => exportFile(format))}>Export .{format}</button>)}
          <button type="button" disabled={busy || live(detail.status) || detail.notes_status === "generating"} onClick={() => setConfirmDelete(true)}>Delete meeting</button>
        </div>
        {confirmDelete && <div className="meeting-delete" role="alert">
          <p>Delete this meeting’s transcript and notes? This cannot be undone. Export a copy first if you need it.</p>
          <button type="button" disabled={busy} onClick={() => void act(async () => {
            await request(`/meetings/${selected}/delete`, { method: "POST", body: "{}" });
            current.current = null; setSelected(null); setDetail(null); localStorage.removeItem(MEETING_SELECTION);
          }, "Meeting deleted.")}>Delete permanently</button>
          <button type="button" onClick={() => setConfirmDelete(false)}>Keep meeting</button>
        </div>}
      </>}
    </article>
  </section>;
}
