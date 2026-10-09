import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import { tintedBone } from "../ui/coatApply";
import type { ResearchJob } from "./useSocket";
import "./research.css";

/** Every research bone gets the next of these, so two jobs never look alike. */
const COLOURS = ["#c6abff", "#65e6bf", "#ffd479", "#ff9fc3", "#8fd3ff"];
const colours = new Map<string, string>();
let nextColour = 0;
function colourOf(id: string) {
  if (!colours.has(id)) colours.set(id, COLOURS[nextColour++ % COLOURS.length]);
  return colours.get(id)!;
}

/** Jobs already flown in, so each flies once. */
const landed = new Set<string>();

const STATUS_LABEL = { working: "Researching", done: "Research ready", failed: "Couldn't search" };

/** Read or failed bones go this long after their card closes. */
const LINGER = 20_000;

function host(url: string) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

/** Pops out of Mellow's head, hops, and arcs over to its parking slot. */
function fly(el: HTMLElement) {
  const head = document.querySelector(".pet-body")?.getBoundingClientRect();
  if (!head || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 220 });
    return;
  }
  const slot = el.getBoundingClientRect();
  const x = head.left + head.width * 0.42 - (slot.left + slot.width / 2);
  const y = head.top + head.height * 0.22 - (slot.top + slot.height / 2);
  const apex = Math.min(y - 70, 0) - 60;
  el.animate(
    [
      { transform: `translate(${x}px, ${y}px) scale(0.3)`, opacity: 0, easing: "cubic-bezier(0.34, 1.56, 0.64, 1)" },
      { transform: `translate(${x}px, ${y - 64}px) scale(1.2)`, opacity: 1, offset: 0.26, easing: "ease-in-out" },
      { transform: `translate(${x}px, ${y - 52}px) scale(1)`, offset: 0.34, easing: "ease-in" },
      { transform: `translate(${x / 2}px, ${apex}px) scale(1.05)`, offset: 0.6, easing: "ease-out" },
      { transform: "translate(0, 0) scale(1)", offset: 0.84 },
      { transform: "translate(0, 3px) scale(1.25, 0.75)", offset: 0.91 },
      { transform: "translate(0, 0) scale(1)" },
    ],
    { duration: 1250 },
  );
}

function Report({ job, onHide, onDismiss, onRetry }: {
  job: ResearchJob;
  onHide: () => void;
  onDismiss: () => void;
  onRetry: () => void;
}) {
  return (
    <div className="panel research-card" data-hit>
      <div className="panel__bar">
        <span className="panel__title">{job.status === "done" ? job.title : STATUS_LABEL[job.status]}</span>
        <span className="panel__actions">
          <button className="panel__btn panel__btn--ghost" onClick={onHide}>Hide</button>
          <button className="panel__btn" onClick={onDismiss}>{job.status === "working" ? "Stop" : "Done"}</button>
        </span>
      </div>
      {job.status === "working" && (
        <>
          <p className="panel__hint">{job.question}</p>
          <div className="research-card__dots" role="status" aria-label="Searching the web">
            <i />
            <i />
            <i />
          </div>
        </>
      )}
      {job.status === "failed" && (
        <>
          <p className="research-card__text">{job.message}</p>
          <p className="panel__hint">{job.question}</p>
          <button className="panel__btn panel__btn--wide" onClick={onRetry}>Try again</button>
        </>
      )}
      {job.status === "done" && (
        <>
          {job.paragraphs?.map((text, i) => <p className="research-card__text" key={i}>{text}</p>)}
          {!!job.sources?.length && (
            <ol className="research-card__sources">
              {job.sources.map((source) => (
                <li key={source.url}>
                  <button type="button" onClick={() => void openUrl(source.url)} title={source.url}>
                    <span>{source.title}</span>
                    <small>{host(source.url)}</small>
                  </button>
                </li>
              ))}
            </ol>
          )}
        </>
      )}
    </div>
  );
}

function ParkedBone({ job, open, onToggle, onDismiss, onRetry }: {
  job: ResearchJob;
  open: boolean;
  onToggle: () => void;
  onDismiss: () => void;
  onRetry: () => void;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  // Whether they have read the finished card. A retry starts the reading over.
  const seen = useRef(false);
  if (job.status === "working") seen.current = false;
  else if (open) seen.current = true;
  const dismiss = useRef(onDismiss);
  dismiss.current = onDismiss;
  useEffect(() => {
    // An unread report stays as the notice; a read one, or a failure, tidies away.
    if (open || job.status === "working" || (job.status === "done" && !seen.current)) return;
    const timer = window.setTimeout(() => dismiss.current(), LINGER);
    return () => window.clearTimeout(timer);
  }, [job.status, open]);
  const fill = colourOf(job.id);
  const [art, setArt] = useState<string>();
  useEffect(() => {
    let live = true;
    void tintedBone(fill).then((url) => live && setArt(url));
    return () => { live = false; };
  }, [fill]);
  useLayoutEffect(() => {
    if (!ref.current || landed.has(job.id)) return;
    landed.add(job.id);
    fly(ref.current);
  }, [job.id]);

  return (
    <div className="research-slot">
      {open && <Report job={job} onHide={onToggle} onDismiss={onDismiss} onRetry={onRetry} />}
      {!open && job.status === "done" && (
        <button type="button" className="research-chip" data-hit onClick={onToggle}>Open</button>
      )}
      {!open && job.status === "failed" && (
        <button type="button" className="research-chip research-chip--failed" data-hit onClick={onToggle}>
          Couldn&rsquo;t search
        </button>
      )}
      <button
        type="button"
        ref={ref}
        className="research-pin"
        data-hit
        data-status={job.status}
        onClick={onToggle}
        title={`${STATUS_LABEL[job.status]}: ${job.question}`}
        aria-label={`${STATUS_LABEL[job.status]}: ${job.question}`}
      >
        <i className="research-bone" style={art ? { backgroundImage: `url(${art})` } : undefined} />
        <i className="research-dot" />
      </button>
    </div>
  );
}

/** Research bones parked in the top-right corner, one per job, each opening its report. */
export function ResearchTray({ jobs, trayRef, onDismiss, onRetry }: {
  jobs: ResearchJob[];
  trayRef: React.RefObject<HTMLDivElement | null>;
  onDismiss: (id: string) => void;
  onRetry: (id: string) => void;
}) {
  const [openId, setOpenId] = useState<string | null>(null);
  if (!jobs.length) return null;
  return (
    <div className="research-tray" ref={trayRef}>
      {jobs.map((job) => (
        <ParkedBone
          key={job.id}
          job={job}
          open={openId === job.id}
          onToggle={() => setOpenId((id) => (id === job.id ? null : job.id))}
          onDismiss={() => {
            setOpenId(null);
            onDismiss(job.id);
          }}
          onRetry={() => onRetry(job.id)}
        />
      ))}
    </div>
  );
}
