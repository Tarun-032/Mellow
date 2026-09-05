import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { request } from "../ui/fields";

export type MeetingStatus = {
  id: string | null;
  status: "idle" | "starting" | "recording" | "paused" | "finalizing" | "complete" | "interrupted";
  active: boolean;
  duration: number;
  warning: string;
  pending: number;
  levels: Record<string, number>;
};

export const MEETING_SELECTION = "mellow-meeting-selection";
export const MEETING_OPEN = "mellow-meetings-open";

export function clock(seconds: number) {
  const s = Math.floor(Math.max(0, seconds));
  return `${String(Math.floor(s / 3600)).padStart(2, "0")}:${String(Math.floor(s / 60) % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

export async function viewMeeting(id: string | null) {
  if (id) localStorage.setItem(MEETING_SELECTION, id);
  localStorage.setItem(MEETING_OPEN, "1");
  await invoke("open_meetings");
}

export function useMeeting() {
  const [status, setStatus] = useState<MeetingStatus | null>(null);
  const [error, setError] = useState("");
  const refresh = useCallback(async () => {
    const next = await request<MeetingStatus>("/meetings/status");
    setStatus(next);
    setError("");
    return next;
  }, []);
  // Pause/resume/stop straight from the compact badge, no panel needed.
  const control = useCallback(async (action: "pause" | "resume" | "stop") => {
    await request(`/meetings/${action}`, { method: "POST", body: "{}" });
    await refresh();
  }, [refresh]);
  useEffect(() => {
    let alive = true;
    let timer = 0;
    const poll = async () => {
      try {
        const next = await request<MeetingStatus>("/meetings/status");
        if (alive) { setStatus(next); setError(""); }
      } catch {
        if (alive) setError("Cannot reach Mellow's helper. Check that Mellow is running. Recording status is unknown.");
      }
      if (alive) timer = window.setTimeout(poll, 750);
    };
    void poll();
    return () => { alive = false; clearTimeout(timer); };
  }, []);
  return { status, error, refresh, control };
}
