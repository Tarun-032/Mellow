import { useEffect, useState } from "react";
import type { WritingStatus } from "./useSocket";

export function WritingPanel({ draft, send }: { draft: WritingStatus; send: (message: object) => void }) {
  const [notice, setNotice] = useState("");
  const [retrying, setRetrying] = useState(false);
  useEffect(() => { setNotice(""); setRetrying(false); }, [draft]);
  const working = draft.status === "thinking" || draft.status === "inserting";
  const dismiss = () => send({ type: "writing_dismiss", id: draft.id });
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(draft.text);
      setNotice("Draft copied.");
    } catch {
      setNotice("Clipboard unavailable. Try copying again.");
    }
  };
  return (
    <section className="panel writing-panel" aria-label="Mellow writing"
      onPointerDown={(event) => { event.preventDefault(); event.stopPropagation(); }}
      onContextMenu={(event) => { event.preventDefault(); event.stopPropagation(); }}>
      <div className="panel__bar">
        <span className="panel__title">{working ? "Writing" : "Your draft"}</span>
        <button className="panel__btn panel__btn--ghost" type="button" onClick={dismiss}>
          {working ? "Cancel" : "Dismiss"}
        </button>
      </div>
      <p className="panel__hint" role="status">{notice || draft.message}</p>
      {draft.text && <pre className="writing-panel__text">{draft.text}</pre>}
      {!working && draft.text && (
        <div className="panel__row writing-panel__actions">
          <button className="panel__btn" type="button" onClick={() => void copy()}>Copy draft</button>
          {draft.retry && <button className="panel__btn panel__btn--ghost" type="button" disabled={retrying}
            onClick={() => { setRetrying(true); send({ type: "writing_retry", id: draft.id }); }}>
            {retrying ? "Checking field…" : "Retry insertion"}
          </button>}
        </div>
      )}
      {draft.retry && <p className="panel__hint">Click the original field first, then Retry insertion.</p>}
    </section>
  );
}
