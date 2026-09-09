import { getVersion } from "@tauri-apps/api/app";
import { check, type Update } from "@tauri-apps/plugin-updater";

export const UPDATE_NOTICE_KEY = "mellow-update-noticed";

/** Where to send someone whose check failed. */
export const RELEASES_URL = "https://github.com/Tarun-032/Mellow/releases";

export type UpdateProgress = {
  downloaded: number;
  total: number | null;
  finished: boolean;
};

export const installedVersion = () => getVersion();

/**
 * Null when this build is current. No fallback on purpose: the old one asked the
 * GitHub API when the endpoint 404'd, could only answer "up to date", and so
 * reported a real update as a connection error.
 */
export function findUpdate() {
  return check({ timeout: 15_000 });
}

export async function installUpdate(
  update: Update,
  onProgress: (progress: UpdateProgress) => void,
) {
  let downloaded = 0;
  let total: number | null = null;

  // Does not return on Windows: the plugin exits the process after handing off
  // to the installer, which reopens Mellow itself.
  await update.downloadAndInstall(
    (event) => {
      if (event.event === "Started") {
        total = event.data.contentLength ?? null;
        onProgress({ downloaded, total, finished: false });
      } else if (event.event === "Progress") {
        downloaded += event.data.chunkLength;
        onProgress({ downloaded, total, finished: false });
      } else {
        onProgress({ downloaded: total ?? downloaded, total, finished: true });
      }
    },
    { timeout: 600_000, restartAfterInstall: true },
  );
}

export function updateErrorMessage(action: "check" | "install") {
  return action === "check"
    ? "Mellow couldn’t check for updates. Check your connection, or open the releases page."
    : "The update couldn’t be installed. Mellow is unchanged, so you can safely try again.";
}
