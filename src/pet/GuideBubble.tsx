import { listen } from "@tauri-apps/api/event";
import { useEffect, useState } from "react";
import {
  GUIDE_DIALOGUE_KEY,
  readGuideDialogue,
  type GuideDialogue,
} from "./guideDialogue";
import "./guideBubble.css";

const EMPTY: GuideDialogue = {
  text: "",
  error: false,
  side: "left",
  lift: "above",
};

/** Cross-monitor pointing explanation window. */
export default function GuideBubble() {
  const [dialogue, setDialogue] = useState(() => readGuideDialogue() ?? EMPTY);

  useEffect(() => {
    const stop = listen<GuideDialogue>("guide-dialogue", ({ payload }) => {
      setDialogue(payload);
    });
    const stored = (event: StorageEvent) => {
      if (event.key !== GUIDE_DIALOGUE_KEY) return;
      const current = readGuideDialogue();
      if (current) setDialogue(current);
    };
    window.addEventListener("storage", stored);
    return () => {
      stop.then((off) => off()).catch(() => {});
      window.removeEventListener("storage", stored);
    };
  }, []);

  return (
    <main className={`guide-dialogue is-${dialogue.side} is-${dialogue.lift}`}>
      <div className="guide-dialogue__bubble">
        <div
          className={`guide-dialogue__text${dialogue.error ? " is-error" : ""}`}
        >
          {dialogue.text}
        </div>
      </div>
    </main>
  );
}
