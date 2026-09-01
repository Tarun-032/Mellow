import React from "react";
import ReactDOM from "react-dom/client";
import { getCurrentWindow } from "@tauri-apps/api/window";

const label = getCurrentWindow().label;

const entry =
  label === "settings"
    ? import("./settings/Settings")
    : label === "welcome"
      ? import("./onboarding/Onboarding")
      : label === "guide"
        ? import("./pet/Guide")
        : label === "guide-bubble"
          ? import("./pet/GuideBubble")
        : import("./pet/Pet");

entry
  .then(({ default: App }) => {
    ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
      <React.StrictMode>
        <App />
      </React.StrictMode>,
    );
  })
  .catch((error) => console.error("[mellow] could not load window:", error));
