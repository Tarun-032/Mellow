import { useCallback, useEffect, useRef, useState } from "react";
import { request } from "../ui/fields";
import { Confirm } from "../meetings/Meetings";
import "./memory.css";

export type Profile = { nickname: string; occupation: string; about: string };

type Section = { title: string; text: string };
type MemoryView = {
  enabled: boolean;
  document: Section[];
  updated: string;
  pending: { id: number; text: string }[];
  limit: number;
  learning_queued: number;
  remember_conversations: boolean;
};

/** "5 minutes ago", "yesterday", "3 days ago". */
function ago(ts: string): string {
  const seconds = Math.max(0, (Date.now() - new Date(ts).getTime()) / 1000);
  if (seconds < 60) return "just now";
  const units: Array<[number, string]> = [[86400, "day"], [3600, "hour"], [60, "minute"]];
  for (const [size, name] of units) {
    const n = Math.floor(seconds / size);
    if (n >= 1) return n === 1 && name === "day" ? "yesterday" : `${n} ${name}${n === 1 ? "" : "s"} ago`;
  }
  return "just now";
}

const learning = (view: MemoryView) => view.enabled && view.learning_queued > 0;
const chats = (n: number) => `${n} chat${n === 1 ? "" : "s"}`;

/** The same Markdown the sidecar renders for the prompt: "## Title" then the paragraph. */
const rendered = (sections: Section[]) =>
  sections.filter(s => s.text.trim()).map(s => `## ${s.title}\n${s.text}`).join("\n\n");

function MoreMenu({ items }: { items: Array<{ label: string; danger?: boolean; onSelect: () => void }> }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const outside = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    const escape = (e: KeyboardEvent) => { if (e.key === "Escape") { e.stopPropagation(); setOpen(false); } };
    document.addEventListener("mousedown", outside);
    document.addEventListener("keydown", escape, true);
    return () => {
      document.removeEventListener("mousedown", outside);
      document.removeEventListener("keydown", escape, true);
    };
  }, [open]);
  return <div className="memory-menu" ref={ref}>
    <button type="button" className="memory-dialog__icon" aria-label="More options" aria-haspopup="menu"
      aria-expanded={open} onClick={() => setOpen(v => !v)}>⋯</button>
    {open && <div className="memory-menu__list" role="menu">
      {items.map(item => <button key={item.label} type="button" role="menuitem"
        className={item.danger ? "memory-menu__item memory-menu__item--danger" : "memory-menu__item"}
        onClick={() => { setOpen(false); item.onSelect(); }}>{item.label}</button>)}
    </div>}
  </div>;
}

function SummaryDialog({ view, busy, saveDocument, forget, close }: {
  view: MemoryView; busy: boolean;
  saveDocument: (sections: Section[]) => Promise<boolean>;
  forget: () => Promise<void>;
  close: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [editing, setEditing] = useState<Section[] | null>(null);
  const [confirmForget, setConfirmForget] = useState(false);
  useEffect(() => { ref.current?.showModal(); }, []);
  const reading = learning(view);
  const startEdit = () => setEditing(view.document.length ? view.document.map(s => ({ ...s })) : [{ title: "About you", text: "" }]);
  const change = (i: number, patch: Partial<Section>) =>
    setEditing(current => current && current.map((s, j) => (j === i ? { ...s, ...patch } : s)));
  const used = editing ? rendered(editing).length : 0;

  const empty = !view.enabled
    ? <div className="memory-empty">
      <h3>Memory is off</h3>
      <p>Turn on memory and Mellow will learn from your chats what you like and how you like answers.</p>
    </div>
    : reading
      ? <div className="memory-empty" role="status">
        <span className="meeting-spinner" aria-hidden="true" />
        <h3>Reading your chats</h3>
        <p>Mellow is going through {chats(view.learning_queued)} and writing down what matters to you.
          This usually takes a minute or two.</p>
      </div>
      : <div className="memory-empty">
        <h3>No memories yet</h3>
        <p>Start a conversation. As you talk, Mellow picks up what matters to you, like your work, your
          tastes and how you like answers, and writes it here.</p>
        <button type="button" className="button button--secondary" onClick={startEdit}>Write your own</button>
      </div>;

  return <dialog ref={ref} className="meeting-dialog memory-dialog"
    onCancel={e => { e.preventDefault(); if (editing) setEditing(null); else close(); }}>
    <header className="memory-dialog__head">
      <h2>Memory summary</h2>
      {!editing && view.document.length > 0 && (reading
        ? <span className="memory-live" role="status">Updating from {chats(view.learning_queued)}</span>
        : view.updated && <span>Updated {ago(view.updated)}</span>)}
      <div className="memory-dialog__tools">
        {!editing && <MoreMenu items={[
          { label: "Edit", onSelect: startEdit },
          { label: "Delete and turn off memory", danger: true, onSelect: () => setConfirmForget(true) },
        ]} />}
        <button type="button" className="memory-dialog__icon" aria-label="Close" onClick={close}>×</button>
      </div>
    </header>

    <div className="memory-dialog__body">
      {editing ? <div className="memory-edit">
        {editing.map((section, i) => <div key={i} className="memory-edit__section">
          <input className="memory-edit__title" maxLength={40} value={section.title} aria-label="Section title"
            placeholder="Section title" onChange={e => change(i, { title: e.target.value })} />
          <button type="button" className="memory-dialog__icon memory-edit__remove"
            aria-label={`Remove ${section.title || "section"}`}
            onClick={() => setEditing(current => current && current.filter((_, j) => j !== i))}>×</button>
          <textarea className="memory-edit__text" value={section.text} aria-label={`${section.title || "Section"} text`}
            placeholder="Write what Mellow should remember, in a sentence or two."
            onChange={e => change(i, { text: e.target.value })} />
        </div>)}
        <button type="button" className="button button--quiet memory-edit__add" disabled={editing.length >= 8}
          onClick={() => setEditing(current => current && [...current, { title: "", text: "" }])}>+ Add section</button>
      </div> : view.document.length ? <div className="memory-summary__text">
        {view.document.map(section => <section key={section.title}>
          <h3>{section.title}</h3>
          <p>{section.text}</p>
        </section>)}
      </div> : empty}
      {!editing && view.enabled && view.pending.length > 0 && <p className="memory-pending">
        <b>Just told Mellow:</b> {view.pending.map(p => p.text).join(" · ")}
        <span> These are already used in answers and will be added to your summary shortly.</span>
      </p>}
    </div>

    {editing && <footer className="memory-dialog__foot">
      <span className={used > view.limit ? "memory-count memory-count--over" : "memory-count"}>
        {used.toLocaleString()} / {view.limit.toLocaleString()} characters
      </span>
      <div>
        <button type="button" className="button button--quiet" onClick={() => setEditing(null)}>Cancel</button>
        <button type="button" className="button button--secondary" disabled={busy || used > view.limit}
          onClick={() => void saveDocument(editing).then(ok => { if (ok) setEditing(null); })}>Save</button>
      </div>
    </footer>}

    {confirmForget && <Confirm busy={busy}
      heading="Delete your memory and turn it off?"
      body="Your memory summary is deleted and Mellow stops remembering. This also starts a new conversation. Your saved chats stay in Sessions; if you turn memory back on, Mellow learns from them again unless you delete them first."
      confirmLabel="Delete and turn off"
      onCancel={() => setConfirmForget(false)}
      onConfirm={() => { void forget().then(() => { setConfirmForget(false); close(); }); }} />}
  </dialog>;
}

export default function Personalization({ prompt, onPrompt, defaultPrompt, profile, onProfile }: {
  prompt: string;
  onPrompt: (value: string) => void;
  defaultPrompt: string;
  profile: Profile;
  onProfile: (value: Profile) => void;
}) {
  const [view, setView] = useState<MemoryView | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [managing, setManaging] = useState(false);

  const act = useCallback(async (work: () => Promise<MemoryView>) => {
    setBusy(true);
    setError("");
    try {
      setView(await work());
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setBusy(false);
    }
  }, []);
  const load = useCallback(() => act(() => request<MemoryView>("/memory")), [act]);
  useEffect(() => {
    void load();
    const refresh = () => void load();
    window.addEventListener("focus", refresh);
    return () => window.removeEventListener("focus", refresh);
  }, [load]);
  // While chats are being learned, show the summary fill in.
  const reading = !!view && learning(view);
  useEffect(() => {
    if (!reading) return;
    const timer = window.setInterval(() => {
      void request<MemoryView>("/memory").then(setView).catch(() => undefined);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [reading]);

  const saveDocument = (sections: Section[]) =>
    act(() => request<MemoryView>("/memory/document", { method: "PUT", body: JSON.stringify({ sections }) }));
  const forget = async () => {
    await act(() => request<MemoryView>("/memory/clear", { method: "POST", body: JSON.stringify({ disable: true }) }));
  };
  const field = (key: keyof Profile) => (value: string) => onProfile({ ...profile, [key]: value });

  // Only what is true right now; an empty memory says nothing here, the window explains it.
  const summaryLine = !view || !view.enabled ? ""
    : reading ? `Reading ${chats(view.learning_queued)}…`
      : view.document.length && view.updated ? `Updated ${ago(view.updated)}`
        : view.pending.length ? `${view.pending.length} thing${view.pending.length === 1 ? "" : "s"} waiting to be added`
          : "";

  return <section className="settings-page personalization" aria-labelledby="personalization-heading">
    <div className="page-heading">
      <h2 id="personalization-heading">Make Mellow yours</h2>
      <p>Tell Mellow who you are and how to respond, and let it learn the rest from your chats.</p>
    </div>

    <div className="settings-group">
      <h3 className="personalization__title">About you</h3>
      <label>
        Nickname
        <input maxLength={60} value={profile.nickname} onChange={e => field("nickname")(e.target.value)}
          placeholder="What should Mellow call you?" />
      </label>
      <label>
        Occupation
        <input maxLength={120} value={profile.occupation} onChange={e => field("occupation")(e.target.value)}
          placeholder="Student, nurse, software engineer" />
      </label>
      <label>
        More about you
        <textarea className="prompt-box personalization__about" maxLength={1500} value={profile.about}
          onChange={e => field("about")(e.target.value)}
          placeholder="Interests, values, or preferences to keep in mind" />
      </label>
    </div>

    <div className="settings-group memory-setting">
      <h3 className="personalization__title">Memory</h3>
      <label className="switch">
        <span className="switch__text">
          <b>Enable memory</b>
          <small>Let Mellow learn from your chats and give you personalized replies that work the way you want.</small>
        </span>
        <input type="checkbox" role="switch" checked={view?.enabled ?? false} disabled={busy || !view}
          onChange={e => {
            const enabled = e.target.checked;
            void act(() => request<MemoryView>("/memory/settings", { method: "PUT", body: JSON.stringify({ enabled }) }));
          }} />
        <i className="switch__track" aria-hidden="true" />
      </label>
      <div className="memory-row">
        <span className="switch__text">
          <b>Memory summary</b>
          <small>View and edit what Mellow has learned about you.</small>
          {summaryLine && <small className={reading ? "memory-row__meta memory-live" : "memory-row__meta"}
            role={reading ? "status" : undefined}>{summaryLine}</small>}
        </span>
        <button type="button" className="button button--secondary" disabled={!view}
          onClick={() => setManaging(true)}>Manage</button>
      </div>
      {view?.enabled && !view.remember_conversations &&
        <p className="memory-status">Learning from chats needs Sessions, Remember sessions.</p>}
      {error && <p className="notice notice--error" role="alert">{error}</p>}
    </div>

    <div className="settings-group">
      <h3 className="personalization__title">Personality</h3>
      <label>
        How should Mellow respond? <span className="optional">optional</span>
        <textarea className="prompt-box" value={prompt} spellCheck={false}
          onChange={e => onPrompt(e.target.value)}
          placeholder="For example: be a little playful, and keep answers to two sentences" />
      </label>
      <button type="button" className="button button--quiet prompt-reset" disabled={!prompt}
        onClick={() => onPrompt(defaultPrompt)}>Clear instructions</button>
    </div>

    {managing && view && <SummaryDialog view={view} busy={busy} saveDocument={saveDocument}
      forget={forget} close={() => { setManaging(false); void load(); }} />}
  </section>;
}
