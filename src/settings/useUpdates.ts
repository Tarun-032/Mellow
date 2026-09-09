import { useCallback, useEffect, useRef, useState } from "react";
import type { Update } from "@tauri-apps/plugin-updater";
import {
  findUpdate,
  installedVersion,
  installUpdate,
  updateErrorMessage,
  type UpdateProgress,
} from "../updater";

export type UpdateStatus =
  | "idle"
  | "checking"
  | "current"
  | "available"
  | "downloading"
  | "installing"
  | "error";

export type UpdatesController = {
  currentVersion: string;
  availableVersion: string | null;
  notes: string;
  status: UpdateStatus;
  progress: UpdateProgress | null;
  error: string;
  check: () => Promise<void>;
  install: () => Promise<void>;
};

export function useUpdates(): UpdatesController {
  const updateRef = useRef<Update | null>(null);
  const checkingRef = useRef(false);
  const installingRef = useRef(false);
  const mountedRef = useRef(false);
  const [currentVersion, setCurrentVersion] = useState("—");
  const [availableVersion, setAvailableVersion] = useState<string | null>(null);
  const [notes, setNotes] = useState("");
  const [status, setStatus] = useState<UpdateStatus>("idle");
  const [progress, setProgress] = useState<UpdateProgress | null>(null);
  const [error, setError] = useState("");

  const checkForUpdates = useCallback(async () => {
    if (checkingRef.current) return;
    checkingRef.current = true;
    setStatus("checking");
    setError("");
    setProgress(null);

    try {
      const version = await installedVersion();
      if (mountedRef.current) setCurrentVersion(version);

      const previous = updateRef.current;
      updateRef.current = null;
      if (previous) await previous.close().catch(() => undefined);

      const update = await findUpdate();
      if (!mountedRef.current) {
        await update?.close().catch(() => undefined);
        return;
      }
      updateRef.current = update;
      setAvailableVersion(update?.version ?? null);
      setNotes(update?.body?.trim() ?? "");
      setStatus(update ? "available" : "current");
    } catch {
      if (mountedRef.current) {
        setAvailableVersion(null);
        setStatus("error");
        setError(updateErrorMessage("check"));
      }
    } finally {
      checkingRef.current = false;
    }
  }, []);

  const install = useCallback(async () => {
    const update = updateRef.current;
    if (!update || status === "downloading" || status === "installing") return;
    setStatus("downloading");
    installingRef.current = true;
    setError("");
    setProgress({ downloaded: 0, total: null, finished: false });

    try {
      await installUpdate(update, (next) => {
        if (!mountedRef.current) return;
        setProgress(next);
        if (next.finished) setStatus("installing");
      });
    } catch {
      if (mountedRef.current) {
        await update.close().catch(() => undefined);
        updateRef.current = null;
        setAvailableVersion(null);
        setStatus("error");
        setError(updateErrorMessage("install"));
      }
    } finally {
      installingRef.current = false;
    }
  }, [status]);

  useEffect(() => {
    mountedRef.current = true;
    void checkForUpdates();
    return () => {
      mountedRef.current = false;
      const update = updateRef.current;
      updateRef.current = null;
      if (update && !installingRef.current) void update.close().catch(() => undefined);
    };
  }, [checkForUpdates]);

  return {
    currentVersion,
    availableVersion,
    notes,
    status,
    progress,
    error,
    check: checkForUpdates,
    install,
  };
}
