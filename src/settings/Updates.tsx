import { openUrl } from "@tauri-apps/plugin-opener";
import { RELEASES_URL } from "../updater";
import type { UpdatesController } from "./useUpdates";

type Props = {
  updates: UpdatesController;
};

const size = (bytes: number) => {
  if (bytes < 1_000_000) return `${Math.round(bytes / 1_000)} KB`;
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
};

export default function Updates({ updates }: Props) {
  const {
    currentVersion,
    availableVersion,
    notes,
    status,
    progress,
    error,
  } = updates;
  const busy = status === "checking" || status === "downloading" || status === "installing";
  const percent = progress?.total
    ? Math.min(100, Math.round((progress.downloaded / progress.total) * 100))
    : null;

  return (
    <section className="settings-page" aria-labelledby="updates-heading">
      <div className="page-heading">
        <h2 id="updates-heading">Keep Mellow current</h2>
        <p>Get the latest version of Mellow.</p>
      </div>

      <div className="settings-group update-panel" aria-live="polite">
        <div className="update-panel__topline">
          <div>
            <span className="update-panel__label">Installed version</span>
            <strong>Mellow {currentVersion}</strong>
          </div>
          <span className={`update-status update-status--${status}`}>
            {status === "available" ? "Update available" : status === "current" ? "Up to date" : "Updates"}
          </span>
        </div>

        {status === "checking" && (
          <div className="update-message">
            <span className="update-spinner" aria-hidden="true" />
            <div>
              <h3>Looking for updates…</h3>
              <p>This usually takes only a moment.</p>
            </div>
          </div>
        )}

        {status === "current" && (
          <div className="update-message">
            <span className="update-check" aria-hidden="true" />
            <div>
              <h3>No update available</h3>
              <p>You are using the latest version of Mellow.</p>
            </div>
          </div>
        )}

        {status === "available" && availableVersion && (
          <div className="update-message update-message--available">
            <div>
              <h3>Mellow {availableVersion} is ready</h3>
              <p>The update will close Mellow, install, and open it again.</p>
              {notes && <p className="update-notes">{notes}</p>}
            </div>
          </div>
        )}

        {(status === "downloading" || status === "installing") && (
          <div className="update-download">
            <div className="update-download__copy">
              <strong>{status === "installing" ? "Installing update…" : "Downloading update…"}</strong>
              <span>
                {status === "installing"
                  ? "Mellow will restart when it is ready."
                  : percent !== null
                    ? `${percent}% · ${size(progress?.downloaded ?? 0)} of ${size(progress?.total ?? 0)}`
                    : size(progress?.downloaded ?? 0)}
              </span>
            </div>
            <progress max={progress?.total ?? undefined} value={progress?.total ? progress.downloaded : undefined} />
          </div>
        )}

        {status === "error" && (
          <div className="notice notice--error update-error" role="alert">
            {error}
          </div>
        )}

        <div className="update-panel__actions">
          {status === "error" && (
            <button
              className="button button--secondary"
              type="button"
              onClick={() => void openUrl(RELEASES_URL)}
            >
              View releases
            </button>
          )}
          {status === "available" ? (
            <button className="button button--primary" type="button" onClick={() => void updates.install()}>
              Update and restart
            </button>
          ) : status !== "downloading" && status !== "installing" ? (
            <button className="button button--secondary" type="button" disabled={busy} onClick={() => void updates.check()}>
              {status === "checking" ? "Checking…" : status === "error" ? "Try again" : "Check again"}
            </button>
          ) : null}
        </div>
      </div>
    </section>
  );
}
