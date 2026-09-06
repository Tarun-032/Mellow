import { useCallback, useEffect, useRef, useState } from "react";
import { request } from "../ui/fields";
import { clock, MEETING_SELECTION, useMeeting } from "./useMeeting";
import { inlineSpans, parseNotes } from "./notes";
import { buildExport, type Content, type Format } from "./export";
import "./meetings.css";

type Summary = { id: string; title: string; created: string; duration: number; status: string; warning: string; notes_status: string };
type Segment = { id: number; start: number; end: number; speaker: string; text: string };
type Detail = Summary & {
  segments: Segment[];
  notes: string; notes_status: string; notes_error: string; notes_progress: string; engine: string;
};
const live = (status: string) => ["starting", "recording", "paused", "finalizing"].includes(status);
const speakerLabel = (speaker: string) => speaker === "Other participants" ? "Other participant" : speaker;
/** `warning` also carries pause disclosure (mellowd/meetings.py). That is normal, not a problem. */
const PAUSE_NOTE = "This meeting includes pauses.";
const EXPORT_MENU = "meeting-export-menu";

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

const spans = (text: string) => inlineSpans(text).map((span, i) =>
  span.bold ? <strong key={i}>{span.text}</strong>
    : span.time ? <span key={i} className="meeting-notes__time">{span.text}</span>
      : <span key={i}>{span.text}</span>);

const PencilIcon = () =>
  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20h4L19 9a2.8 2.8 0 0 0-4-4L4 16v4Z" /><path d="M14 6l4 4" /></svg>;

/** Native <dialog>: Escape, focus trap and the top layer come from the platform. */
function Confirm({ heading, body, confirmLabel, onConfirm, onCancel, busy }: {
  heading: string; body: string; confirmLabel: string;
  onConfirm: () => void; onCancel: () => void; busy: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { ref.current?.showModal(); }, []);
  return <dialog ref={ref} className="meeting-dialog"
    onCancel={e => { e.preventDefault(); onCancel(); }}
    onClick={e => { if (e.target === ref.current) onCancel(); }}>
    <h2>{heading}</h2>
    <p>{body}</p>
    <div className="meeting-dialog__actions">
      <button type="button" className="button button--secondary" autoFocus onClick={onCancel}>Cancel</button>
      <button type="button" className="button button--secondary button--danger" disabled={busy} onClick={onConfirm}>{confirmLabel}</button>
    </div>
  </dialog>;
}

function NotesBody({ markdown }: { markdown: string }) {
  return <div className="meeting-notes">{parseNotes(markdown).map((block, i) => {
    if (block.kind === "heading") return <h3 key={i}>{spans(block.text)}</h3>;
    if (block.kind === "para") return <p key={i}>{spans(block.text)}</p>;
    const items = block.items.map((item, j) => <li key={j} data-depth={item.depth}>{spans(item.text)}</li>);
    return block.ordered ? <ol key={i}>{items}</ol> : <ul key={i}>{items}</ul>;
  })}</div>;
}

export default function Meetings({ openLast = false }: { openLast?: boolean }) {
  const [items, setItems] = useState<Summary[]>([]);
  // The list is the landing page. Settings remounts this component with openLast
  // set only when Mellow asked for one specific meeting.
  const [selected, setSelected] = useState<string | null>(() => openLast ? localStorage.getItem(MEETING_SELECTION) : null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [tab, setTab] = useState<"transcript" | "notes">("transcript");
  const [query, setQuery] = useState("");
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [engine, setEngine] = useState<{ ready: boolean; name: string } | null>(null);
  const [selecting, setSelecting] = useState(false);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [confirmBulk, setConfirmBulk] = useState(false);
  const [pick, setPick] = useState<Content | null>(null);
  const [regenerating, setRegenerating] = useState(false);
  const meeting = useMeeting();
  const headingRef = useRef<HTMLInputElement>(null);
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
    request<{ settings: { ai_enabled: boolean; llm: { mode: string; provider: string; model: string } } }>("/config").then(({ settings }) => {
      const llm = settings.llm;
      setEngine({
        ready: settings.ai_enabled,
        name: llm.mode === "local" ? llm.model : `${llm.provider} (${llm.model || "agent default"})`,
      });
    }).catch(e => setError(e.message));
    return () => { alive = false; clearTimeout(timer); };
  }, [refresh]);

  useEffect(() => {
    setDetail(null); setConfirmDelete(false); setError(""); setNotice("");
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
  // togglePopover(false) over hidePopover(): the latter throws if already closed.
  const closeMenu = () => { document.getElementById(EXPORT_MENU)?.togglePopover(false); setPick(null); };
  const saveFile = (content: Content, format: Format) => {
    if (!detail) return;
    const { text, filename, mime } = buildExport(detail, turns.map(t => ({ speaker: speakerLabel(t.speaker), text: t.text })), content, format);
    const url = URL.createObjectURL(new Blob([text], { type: mime }));
    const link = document.createElement("a"); link.href = url; link.download = filename;
    document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    closeMenu(); setNotice(`Exported ${filename}.`);
  };
  const copyOut = (content: Content) => {
    if (!detail) return;
    const { text } = buildExport(detail, turns.map(t => ({ speaker: speakerLabel(t.speaker), text: t.text })), content, "txt");
    closeMenu();
    void act(() => navigator.clipboard.writeText(text), `${content === "notes" ? "Notes" : "Transcript"} copied to clipboard.`);
  };
  // The server flips notes_status to "generating" before this POST returns, so once
  // act() has refreshed, status alone carries the state. The flag only covers the
  // round trip, and clears if the request is refused so the old notes come back.
  const generate = async () => {
    setRegenerating(true);
    try { await act(() => request(`/meetings/${selected}/notes`, { method: "POST", body: "{}" })); }
    finally { setRegenerating(false); }
  };
  const leaveSelect = () => { setSelecting(false); setPicked(new Set()); setConfirmBulk(false); };
  // Not act(): the message depends on how many actually went, and act() takes a
  // fixed string. One id per request — there is no bulk route, and a recording or
  // notes-generating meeting answers 409, so count what landed and say so.
  const deletePicked = async () => {
    setBusy(true); setError(""); setNotice("");
    let gone = 0;
    const kept: string[] = [];
    for (const id of picked) {
      try {
        await request(`/meetings/${id}/delete`, { method: "POST", body: "{}" });
        gone++;
        if (localStorage.getItem(MEETING_SELECTION) === id) localStorage.removeItem(MEETING_SELECTION);
      } catch { kept.push(id); }
    }
    leaveSelect();
    try { await refresh(); } catch { /* the 2s poll picks it up */ }
    setBusy(false);
    if (kept.length) setError(`Deleted ${gone}. ${kept.length} could not be deleted. Stop recording and let notes generation finish first.`);
  };
  const filtered = items.filter(item => item.title.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
  const turns = conversationTurns(detail?.segments ?? []);

  const banner = <>
    {meeting.status?.active && <div className="meetings-live" role="status">
      <span>Meeting {meeting.status.status} · {clock(meeting.status.duration)}</span>
      <button type="button" className="button button--secondary" onClick={() => setSelected(meeting.status!.id)}>View active meeting</button>
      <button type="button" className="button button--secondary" disabled={busy || meeting.status.status === "finalizing" || meeting.status.status === "starting"}
        onClick={() => void act(() => request("/meetings/stop", { method: "POST", body: "{}" }))}>Stop &amp; save</button>
    </div>}
    {(error || meeting.error) && <p className="notice notice--error" role="alert">{error || meeting.error}</p>}
    {notice && <p className="notice notice--ok" role="status">{notice}</p>}
  </>;

  if (!selected) return <section className="meetings">
    <p className="meetings-intro">Right-click Mellow and choose Transcribe meeting. Your transcript and notes stay here until you delete them, independently of saved conversations.</p>
    {banner}
    <div className="meetings-toolbar">
      <label className="meetings-find">Find a meeting<input type="search" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search titles" /></label>
      <span className="meetings-toolbar__count">{selecting ? `${picked.size} selected` : `${items.length} meeting${items.length === 1 ? "" : "s"}`}</span>
      {selecting ? <>
        <button type="button" className="button button--secondary button--danger" disabled={busy || !picked.size}
          onClick={() => setConfirmBulk(true)}>Delete</button>
        <button type="button" className="button button--secondary" disabled={busy} onClick={leaveSelect}>Cancel</button>
      </> : <button type="button" className="button button--secondary" disabled={!items.length}
        onClick={() => setSelecting(true)}>Select</button>}
    </div>
    {confirmBulk && <Confirm busy={busy}
      heading={`Delete ${picked.size} meeting${picked.size === 1 ? "" : "s"}?`}
      body={`${picked.size === 1 ? "Its transcript and notes go with it." : "Their transcripts and notes go with them."} This cannot be undone. Export a copy first if you need one.`}
      confirmLabel="Delete permanently" onCancel={() => setConfirmBulk(false)} onConfirm={() => void deletePicked()} />}
    {!loaded ? <p role="status">Loading meetings…</p> : !items.length ? <p>No meetings yet. Start your first from Mellow’s right-click menu.</p> : !filtered.length ? <p>No matching meetings.</p> :
      <ul className={`meetings-list${selecting ? " meetings-list--selecting" : ""}`}>{filtered.map(item => {
        const row = <>
          <span className="meetings-row__title">{item.title}</span>
          <span className="meetings-row__when">{new Date(item.created).toLocaleString()}</span>
        </>;
        // A <label> in select mode so the whole row toggles its checkbox natively.
        return <li key={item.id}>{selecting
          ? <label className="meetings-row">
            <input type="checkbox" checked={picked.has(item.id)} onChange={e => setPicked(was => {
              const next = new Set(was);
              if (e.target.checked) next.add(item.id); else next.delete(item.id);
              return next;
            })} />
            {row}
          </label>
          : <button type="button" className="meetings-row" onClick={() => setSelected(item.id)}>
            {row}<span className="meetings-row__go" aria-hidden="true">›</span>
          </button>}</li>;
      })}</ul>}
  </section>;

  const problem = detail && !detail.warning.startsWith(PAUSE_NOTE) ? detail.warning : "";
  const writing = regenerating || detail?.notes_status === "generating";
  return <section className="meetings">
    <button type="button" className="button button--quiet meetings-back" onClick={() => setSelected(null)}>← All meetings</button>
    {banner}
    <article className="meeting-detail" aria-label="Selected meeting">
      {!detail ? <p role="status">Loading transcript…</p> : <>
        <div className="meeting-title">
          <input ref={headingRef} className="meeting-heading" aria-label="Meeting title" value={title} maxLength={160}
            onChange={e => setTitle(e.target.value)} onKeyDown={e => { if (e.key === "Enter") e.preventDefault(); }} />
          {/* The pencil is what people aim at, so it has to do the job itself. */}
          <button type="button" className="meeting-title__edit" aria-label="Rename this meeting" title="Rename"
            onClick={() => {
              const field = headingRef.current;
              if (!field) return;
              field.focus();
              field.setSelectionRange(field.value.length, field.value.length);
            }}><PencilIcon /></button>
          {title.trim() && title !== detail.title && <button type="button" className="button button--secondary" disabled={busy}
            onClick={() => void act(() => request(`/meetings/${selected}`, { method: "PUT", body: JSON.stringify({ title }) }))}>Save name</button>}
        </div>
        {problem && <p className="notice notice--error" role="alert">{problem}</p>}
        <div className="meeting-bar">
          <div className="meeting-tabs" role="group" aria-label="Meeting view">
            <button type="button" aria-pressed={tab === "transcript"} onClick={() => setTab("transcript")}>Transcript</button>
            <button type="button" aria-pressed={tab === "notes"} onClick={() => setTab("notes")}>Notes</button>
          </div>
          <div className="meeting-bar__actions">
            {tab === "notes" && detail.notes && !writing && <button type="button" className="button button--secondary" onClick={() => void generate()}
              disabled={busy || live(detail.status) || !engine?.ready}>Regenerate notes</button>}
            <button type="button" className="button button--secondary meeting-bar__export" popoverTarget={EXPORT_MENU}>Export ▾</button>
          </div>
        </div>
        {/* Native popover: light dismiss and Escape without a click-outside handler. */}
        <div id={EXPORT_MENU} popover="auto" className="meeting-menu" onToggle={e => { if (e.newState === "closed") setPick(null); }}>
          {!pick ? <>
            <p className="meeting-menu__head">Export what?</p>
            <button type="button" onClick={() => setPick("transcript")} disabled={!turns.length}>Transcript</button>
            <button type="button" onClick={() => setPick("notes")} disabled={!detail.notes}>Notes</button>
          </> : <>
            <p className="meeting-menu__head">
              <button type="button" className="meeting-menu__back" onClick={() => setPick(null)} aria-label="Back to content">←</button>
              {pick === "notes" ? "Notes as" : "Transcript as"}
            </p>
            {([["md", "Markdown (.md)"], ["txt", "Plain text (.txt)"], ["json", "JSON (.json)"]] as [Format, string][])
              .map(([format, label]) => <button type="button" key={format} onClick={() => saveFile(pick, format)}>{label}</button>)}
            <button type="button" className="meeting-menu__copy" onClick={() => copyOut(pick)}>Copy to clipboard</button>
          </>}
        </div>
        {tab === "transcript" ? <>
          <div className="meeting-transcript">
            {!turns.length ? <p>{live(detail.status) ? "Listening. Your conversation appears here as speech is processed." : "No speech was transcribed. Check your microphone, meeting output and speech-to-text settings before recording again."}</p> :
              turns.map(segment => <section key={segment.id}>
                <div><strong>{speakerLabel(segment.speaker)}</strong></div><p>{segment.text}</p>
              </section>)}
          </div>
        </> : <>
          {detail.notes_error && <p className="notice notice--error" role="alert">{detail.notes_error}</p>}
          {/* Writing wins over existing notes: a regenerate should not leave the old
              ones on screen looking current while new ones are being written. */}
          {writing ? <div className="empty-state meeting-notes-empty">
            <span className="meeting-spinner" aria-hidden="true" />
            <h3>{detail.notes ? "Rewriting notes…" : "Writing notes…"}</h3>
            <p role="status">{detail.notes_progress || "Reading the transcript…"} You can leave this page.</p>
          </div> : detail.notes ? <>
            <NotesBody markdown={detail.notes} />
            <p className="meeting-meta">Generated with {detail.engine}. Review these notes against the transcript.</p>
          </> : !engine ? <p role="status">Loading…</p>
            : !engine.ready ? <p className="notice">Choose an answer engine in Settings first, then come back to write notes.</p>
              : <div className="empty-state meeting-notes-empty">
                <h3>No notes yet</h3>
                <p>Mellow reads the transcript and writes an overview, decisions and action items.</p>
                <button type="button" className="button button--primary" onClick={() => void generate()}
                  disabled={busy || live(detail.status) || !detail.segments.length}>Generate notes</button>
                <small>{live(detail.status) ? "Stop recording and let the remaining audio finish first."
                  : !detail.segments.length ? "There is no transcript to summarise yet."
                    : `Processed by ${engine.name}.`}</small>
              </div>}
        </>}
        <div className="meeting-actions">
          <button type="button" className="button button--quiet button--danger meeting-actions__delete" disabled={busy || live(detail.status) || detail.notes_status === "generating"} onClick={() => setConfirmDelete(true)}>Delete meeting</button>
        </div>
        {confirmDelete && <Confirm busy={busy}
          heading="Delete this meeting?"
          body="Its transcript and notes go with it. This cannot be undone. Export a copy first if you need one."
          confirmLabel="Delete permanently" onCancel={() => setConfirmDelete(false)}
          onConfirm={() => void act(async () => {
            await request(`/meetings/${selected}/delete`, { method: "POST", body: "{}" });
            current.current = null; setSelected(null); setDetail(null); localStorage.removeItem(MEETING_SELECTION);
          })} />}
      </>}
    </article>
  </section>;
}
