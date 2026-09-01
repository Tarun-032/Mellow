import { invoke } from "@tauri-apps/api/core";
import { useEffect } from "react";
import sprites from "./sprites.json" with { type: "json" };
import "./guide.css";

const SCALE = 2;
const PADDING = 6;
const SIZE = 36;
const TIP_X = PADDING + sprites.bone.tip.x * SCALE;
const TIP_Y = PADDING + sprites.bone.tip.y * SCALE;

export default function Guide() {
  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

    const ready = () =>
      invoke("guide_ready", {
        width: SIZE,
        height: SIZE,
        tipX: TIP_X,
        tipY: TIP_Y,
        reducedMotion: reduced.matches,
      }).catch((error) => console.error("[mellow] guide setup failed:", error));

    const preferenceChanged = () => {
      void invoke("guide_set_reduced_motion", {
        reducedMotion: reduced.matches,
      }).catch((error) =>
        console.error("[mellow] guide motion preference failed:", error),
      );
    };

    void ready();
    reduced.addEventListener("change", preferenceChanged);
    return () => reduced.removeEventListener("change", preferenceChanged);
  }, []);

  return (
    <main className="guide" aria-hidden="true">
      <i className="guide__bone" />
    </main>
  );
}
