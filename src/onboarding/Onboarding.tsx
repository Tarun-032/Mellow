/** First-run setup wizard; writes config once on the done transition. */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  AgentFields,
  CloudFields,
  Field,
  Mode,
  Notice,
  request,
  type AgentInfo,
  type AgentSpeed,
  type Preset,
  type Transport,
} from "../ui/fields";
import "../pet/sprites.css";
import "./onboarding.css";

type Engine = "agent" | "local" | "key" | "pet";
type Where = "local" | "cloud";

type Screen =
  | "welcome"
  | "engine"
  | "engine-agent"
  | "engine-local"
  | "engine-key"
  | "hearing"
  | "hearing-cloud"
  | "download"
  | "voice"
  | "voice-cloud"
  | "pet-only"
  | "done";

type TestState = {
  state: "idle" | "testing" | "ok" | "failed";
  detail?: string;
};

type Progress = {
  state: "idle" | "running" | "done" | "failed";
  name: string;
  done: number;
  total: number;
  error: string;
};

const IDLE_PROGRESS: Progress = { state: "idle", name: "", done: 0, total: 0, error: "" };

/** Which stage dot a screen lights; the bookends light nothing. */
const STEP_OF: Record<Screen, "engine" | "hearing" | "voice" | "done" | null> = {
  welcome: null,
  engine: "engine",
  "engine-agent": "engine",
  "engine-local": "engine",
  "engine-key": "engine",
  hearing: "hearing",
  "hearing-cloud": "hearing",
  download: "hearing",
  voice: "voice",
  "voice-cloud": "voice",
  "pet-only": "done",
  done: "done",
};

/** Which pose the dog holds, per the design: he works while you wait. */
function poseOf(screen: Screen, downloadSeconds: number): string {
  if (screen === "welcome") return "idle";
  if (screen.startsWith("engine")) return "thinking";
  if (screen.startsWith("hearing")) return "listening";
  if (screen === "download") return downloadSeconds >= 90 ? "sleeping" : "yawn";
  if (screen.startsWith("voice")) return "idle";
  return "petting";
}

/** Working-stage dots (bookends show none). */
const WORKING_STEPS = ["engine", "hearing", "voice"] as const;

/** Welcome capability marks (inline SVG, one word each). */
const CAN_DO: readonly (readonly [string, ReactNode])[] = [
  [
    "Listens",
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="9" y="2" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v4" />
    </svg>,
  ],
  [
    "Looks",
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z" />
      <circle cx="12" cy="12" r="2.6" />
    </svg>,
  ],
  [
    "Speaks",
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3 5h13v10H8l-5 4V5Z" />
      <path d="M19 8a5 5 0 0 1 0 8" />
    </svg>,
  ],
  [
    "Points",
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="7.5" />
      <circle cx="12" cy="12" r="2.25" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
    </svg>,
  ],
];

/** Per-screen title and subtitle. */
const HEADS: Record<Screen, { title: string; sub: string }> = {
  welcome: {
    title: "Meet Mellow",
    sub: "Mellow is a pixel-art desktop companion who listens, speaks, answers questions, and points on your screen to help you get things done.",
  },
  engine: {
    title: "How will Mellow think?",
    sub: "Pick a brain. You can change it any time in Settings.",
  },
  "engine-agent": {
    title: "Connect an agent",
    sub: "One click, one sign in, your subscription does the rest.",
  },
  "engine-local": {
    title: "Pick your model",
    sub: "Ollama runs it here, so nothing leaves this computer.",
  },
  "engine-key": {
    title: "Add a provider",
    sub: "Paste a key from any OpenAI compatible service.",
  },
  hearing: {
    title: "How should Mellow hear?",
    sub: "On device is private. In the cloud is lighter to run.",
  },
  "hearing-cloud": {
    title: "Cloud speech to text",
    sub: "A recording of your voice goes to this provider.",
  },
  download: {
    title: "Getting ears and voice",
    sub: "One time download. Mellow yawns while it runs, and naps if it takes a while.",
  },
  voice: {
    title: "How should Mellow sound?",
    sub: "On device speaks instantly. The cloud offers more voices.",
  },
  "voice-cloud": {
    title: "Cloud text to speech",
    sub: "Hosted voices, fetched once per sentence.",
  },
  "pet-only": {
    title: "Just the pet",
    sub: "No AI. Everything else still works, and it stays that way.",
  },
  done: {
    title: "He is ready",
    sub: "Hold these three keys anywhere, and just talk.",
  },
};

type ConfigResponse = {
  settings: {
    llm: Transport & { vision: string; agent_speed: AgentSpeed };
    stt: Transport & { local_model: string; input_device: string | null };
    tts: Transport & { local_voice: string; voice: string };
    system_prompt: string;
  };
  presets: Record<"llm" | "stt" | "tts", Record<string, Preset>>;
  stt_models: Record<string, string>;
  tts_voices: Record<string, string>;
  default_prompt: string;
};

type Device = { name: string; channels: number; default: boolean };

const KNOWN_SCREENS: Screen[] = [
  // Longer names first: ?screen=engine-agent must match before ?screen=engine.
  "engine-agent", "engine-local", "engine-key", "hearing-cloud", "voice-cloud",
  "welcome", "engine", "hearing", "download", "voice", "pet-only", "done",
];

/** Dev deep link: ?screen=… */
function forcedScreen(): { screen: Screen; variant: string } | null {
  if (!import.meta.env.DEV) return null;
  const raw = new URLSearchParams(window.location.search).get("screen");
  if (!raw) return null;
  for (const base of KNOWN_SCREENS) {
    if (raw === base) return { screen: base, variant: "" };
    if (raw.startsWith(base + "-")) return { screen: base, variant: raw.slice(base.length + 1) };
  }
  return null;
}

export default function Onboarding() {
  const [screen, setScreen] = useState<Screen>(() => forcedScreen()?.screen ?? "welcome");
  const variant = useRef(forcedScreen()?.variant ?? "");
  const [loadError, setLoadError] = useState("");
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [devices, setDevices] = useState<Device[]>([]);
  const [ttsAvailable, setTtsAvailable] = useState(false);

  // The whole wizard's state. Nothing is written until the done screen.
  const [engine, setEngine] = useState<Engine>("agent");
  const [llm, setLlm] = useState<Transport | null>(null);
  const [agentProvider, setAgentProvider] = useState("claude");
  const [agentModel, setAgentModel] = useState("");
  const [agentSpeed, setAgentSpeed] = useState<AgentSpeed>("fast");
  const [hearing, setHearing] = useState<Where>("local");
  const [stt, setStt] = useState<(Transport & { local_model: string; input_device: string | null }) | null>(null);
  const [voice, setVoice] = useState<Where>("local");
  const [tts, setTts] = useState<(Transport & { local_voice: string; voice: string }) | null>(null);

  const [engineTest, setEngineTest] = useState<TestState>({ state: "idle" });
  const [sttTest, setSttTest] = useState<TestState>({ state: "idle" });
  const [ttsTest, setTtsTest] = useState<TestState>({ state: "idle" });
  const [connectState, setConnectState] = useState<TestState>({ state: "idle" });
  const [progress, setProgress] = useState<{ stt: Progress; tts: Progress }>({
    stt: IDLE_PROGRESS,
    tts: IDLE_PROGRESS,
  });
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [downloadSeconds, setDownloadSeconds] = useState(0);

  // ---- Bootstrapping: the sidecar may still be waking up, so retry. ----

  const load = useCallback(() => {
    setLoadError("");
    Promise.all([
      request<ConfigResponse>("/config"),
      request<{ agents: AgentInfo[] }>("/agents"),
      request<{ devices: Device[] }>("/audio/devices"),
      // Voice list is optional; don't block the wizard on it.
      request<{ tts: boolean }>("/models/available").catch(() => ({ tts: false })),
    ])
      .then(([cfg, agentList, audio, available]) => {
        setConfig(cfg);
        setAgents(agentList.agents);
        setDevices(audio.devices);
        setTtsAvailable(available.tts);
        setLlm(cfg.settings.llm);
        setAgentSpeed(cfg.settings.llm.agent_speed);
        setStt(cfg.settings.stt);
        setTts(cfg.settings.tts);
        const installed = agentList.agents.find((a) => a.installed);
        if (installed) setAgentProvider(installed.id);
      })
      .catch(() => setLoadError("not-up"));
  }, []);

  useEffect(() => {
    if (!loadError) return;
    const t = setTimeout(load, 2000);
    return () => clearTimeout(t);
  }, [loadError, load]);

  useEffect(() => {
    load();
  }, [load]);

  // ---- Candidate config: what this wizard would save, right now. ----

  const candidate = useCallback((): Record<string, unknown> | null => {
    if (!config || !llm || !stt || !tts) return null;
    const mode: Mode = engine === "agent" ? "agent" : engine === "key" ? "cloud" : "local";
    return {
      llm: {
        ...llm,
        mode,
        provider: engine === "agent" ? agentProvider : llm.provider,
        model: engine === "agent" ? agentModel : llm.model,
        agent_speed: agentSpeed,
        vision: "auto",
        max_tokens: 4096,
        reasoning_effort: "",
        temperature: 0.3,
      },
      stt: { ...stt, mode: hearing },
      tts: { ...tts, mode: voice, speak: true, speech_speed: 1 },
      system_prompt: config.default_prompt,
      remember_conversations: true,
      ai_enabled: engine !== "pet",
    };
  }, [config, llm, stt, tts, engine, agentProvider, agentModel, agentSpeed, hearing, voice]);

  // ---- Actions ----

  const runEngineTest = async () => {
    const body = candidate();
    if (!body) return;
    setEngineTest({ state: "testing" });
    try {
      const result = await request<{ reply: string }>("/config/test", {
        method: "POST",
        body: JSON.stringify(body),
      });
      setEngineTest({ state: "ok", detail: result.reply });
    } catch (error) {
      setEngineTest({ state: "failed", detail: error instanceof Error ? error.message : String(error) });
    }
  };

  const runSttTest = async () => {
    if (!stt) return;
    setSttTest({ state: "testing" });
    try {
      const result = await request<{ transcript: string; peak: number; device: string }>(
        "/stt/test",
        { method: "POST", body: JSON.stringify({ stt: { ...stt, mode: hearing } }) },
      );
      setSttTest({
        state: "ok",
        detail: result.transcript
          ? `Heard "${result.transcript}" on ${result.device}`
          : `Nothing reached ${result.device}, peak ${result.peak.toFixed(3)}`,
      });
    } catch (error) {
      setSttTest({ state: "failed", detail: error instanceof Error ? error.message : String(error) });
    }
  };

  const runTtsTest = async () => {
    if (!tts) return;
    setTtsTest({ state: "testing" });
    try {
      const result = await request<{ seconds: number; backend: string }>(
        "/tts/test",
        { method: "POST", body: JSON.stringify({ tts: { ...tts, mode: voice, speak: true } }) },
      );
      setTtsTest({ state: "ok", detail: `Spoke ${result.seconds}s with ${result.backend}` });
    } catch (error) {
      setTtsTest({ state: "failed", detail: error instanceof Error ? error.message : String(error) });
    }
  };

  const runConnect = async () => {
    setConnectState({ state: "testing" });
    try {
      const result = await request<{
        ok: boolean;
        signed_in: boolean;
        model_ok: boolean;
        vision_ok: boolean;
        detail: string;
      }>("/agents/login", {
        method: "POST",
        body: JSON.stringify({
          agent: agentProvider,
          model: agentModel,
          agent_speed: agentSpeed,
        }),
      });
      if (result.signed_in && result.ok && result.model_ok && result.vision_ok) {
        setConnectState({ state: "ok", detail: `Signed in and vision verified. ${result.detail}` });
      } else if (result.signed_in) {
        setConnectState({
          state: "failed",
          detail: result.detail || "The selected model failed vision verification.",
        });
      } else {
        setConnectState({
          state: "failed",
          detail: result.detail
            ? `${result.detail}. Sign in, then press Test connection.`
            : "A sign-in window opened. Sign in, close it, then press Test connection.",
        });
      }
    } catch (error) {
      setConnectState({ state: "failed", detail: error instanceof Error ? error.message : String(error) });
    }
  };

  // ---- Downloads ----

  const needsDownload = (which: "stt" | "tts") =>
    (which === "stt" ? hearing : voice) === "local";

  const startDownloads = useCallback(async () => {
    const settings = candidate();
    if (!settings) return;
    for (const which of ["stt", "tts"] as const) {
      if (!needsDownload(which)) continue;
      try {
        await request("/models/download", {
          method: "POST",
          body: JSON.stringify({ which, settings }),
        });
      } catch (error) {
        // Rejected start -> failed (not a stuck 0% bar).
        setProgress((cur) => ({
          ...cur,
          [which]: {
            ...IDLE_PROGRESS,
            state: "failed",
            error: error instanceof Error ? error.message : String(error),
          },
        }));
      }
    }
  }, [candidate, hearing, voice]);

  useEffect(() => {
    if (screen !== "download") return;
    // Dev fake progress: don't start real downloads.
    const dev = import.meta.env.DEV && variant.current !== "";
    if (!dev) void startDownloads();
    const tick = setInterval(async () => {
      setDownloadSeconds((s) => s + 0.5);
      if (dev) return;
      try {
        const p = await request<{ stt: Progress; tts: Progress }>("/models/progress");
        setProgress((cur) => ({
          stt: cur.stt.state === "failed" && p.stt.state === "idle" ? cur.stt : p.stt,
          tts: cur.tts.state === "failed" && p.tts.state === "idle" ? cur.tts : p.tts,
        }));
      } catch {
        /* keep the last known */
      }
    }, 500);
    return () => clearInterval(tick);
  }, [screen, startDownloads]);

  const devProgress = (): { stt: Progress; tts: Progress } | null => {
    if (!import.meta.env.DEV || !variant.current) return null;
    const fake = (state: Progress["state"], done: number, name = "parakeet-tdt-0.6b-v2"): Progress => ({
      state, name, done, total: 640, error: state === "failed" ? "the download failed partway" : "",
    });
    switch (variant.current) {
      case "running": return { stt: fake("running", 210), tts: fake("running", 90) };
      case "nearly": return { stt: fake("running", 600), tts: fake("running", 320) };
      case "failed": return { stt: fake("failed", 300), tts: IDLE_PROGRESS };
      case "done": return { stt: fake("done", 640), tts: fake("done", 337, "kokoro-v1.0.onnx") };
      default: return null;
    }
  };
  const liveProgress = devProgress() ?? progress;

  useEffect(() => {
    if (liveProgress.tts.state === "done") setTtsAvailable(true);
  }, [liveProgress.tts.state]);

  // ---- Derived flow state ----

  const ollamaMissing =
    engineTest.state === "failed" && /11434|can't reach|is it running/.test(engineTest.detail ?? "");
  const agentSelected = agents.find((a) => a.id === agentProvider);
  // Require a successful test before continuing.
  const engineContinueOk = engine === "pet" || engineTest.state === "ok";

  const downloadFailed = (["stt", "tts"] as const).find(
    (w) => needsDownload(w) && liveProgress[w].state === "failed",
  );
  const downloadDone = (["stt", "tts"] as const)
    .filter((w) => needsDownload(w))
    .every((w) => liveProgress[w].state === "done");
  const downloadRunning = (["stt", "tts"] as const)
    .filter((w) => needsDownload(w))
    .some((w) => liveProgress[w].state === "running" || liveProgress[w].state === "idle");
  const overallDone = (["stt", "tts"] as const)
    .filter((w) => needsDownload(w))
    .reduce((sum, w) => {
      const p = liveProgress[w];
      return p.total > 0 ? sum + p.done / p.total : sum;
    }, 0);
  const overallTotal = (["stt", "tts"] as const).filter((w) => needsDownload(w)).length;
  const nearly = overallDone / Math.max(1, overallTotal) >= 0.9;

  // ---- Navigation ----

  const engineDetail: Screen =
    engine === "agent" ? "engine-agent" : engine === "local" ? "engine-local" : "engine-key";

  const goEngine = (choice: Engine) => {
    setEngine(choice);
    setEngineTest({ state: "idle" });
    setConnectState({ state: "idle" });
    if (!config) return;
    if (choice === "pet") return;
    if (choice === "local" && !config.presets.llm[llm?.provider ?? ""]?.local) {
      // Local path forces Ollama before testing.
      patchLlm({ provider: "ollama", base_url: config.presets.llm.ollama.base_url, api_key: "", has_api_key: false, model: "gemma3:4b" });
    }
    if (choice === "key" && config.presets.llm[llm?.provider ?? ""]?.local) {
      const firstCloud = Object.entries(config.presets.llm).find(([, p]) => !p.local);
      if (firstCloud) {
        patchLlm({ provider: firstCloud[0], base_url: firstCloud[1].base_url, api_key: "", has_api_key: false, model: "" });
      }
    }
    setScreen(engineDetailFor(choice));
  };

  const engineDetailFor = (choice: Engine): Screen =>
    choice === "agent" ? "engine-agent" : choice === "local" ? "engine-local" : "engine-key";

  const backFrom = (): Screen | null => {
    switch (screen) {
      case "welcome": return null;
      case "engine": return "welcome";
      case "engine-agent":
      case "engine-local":
      case "engine-key": return "engine";
      case "hearing": return engineDetail;
      case "hearing-cloud": return "hearing";
      case "voice": return hearing === "cloud" ? "hearing-cloud" : "hearing";
      case "voice-cloud": return "voice";
      case "download": return voice === "cloud" ? "voice-cloud" : "voice";
      case "pet-only": return "engine";
      case "done":
        return needsDownload("stt") || needsDownload("tts")
          ? "download"
          : voice === "cloud" ? "voice-cloud" : "voice";
    }
  };

  /** Save config; native completion stays on the final button. */
  const finish = async () => {
    const body = candidate();
    if (!body) return;
    setSaving(true);
    setSaveError("");
    try {
      await request("/config", { method: "PUT", body: JSON.stringify(body) });
      setScreen("done");
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };

  const advance = async () => {
    switch (screen) {
      case "welcome": setScreen("engine"); return;
      case "engine":
        if (engine === "pet") { setScreen("pet-only"); return; }
        setScreen(engineDetail);
        return;
      case "engine-agent":
      case "engine-local":
      case "engine-key": setScreen("hearing"); return;
      case "hearing":
        setScreen(hearing === "cloud" ? "hearing-cloud" : "voice");
        return;
      case "hearing-cloud": setScreen("voice"); return;
      case "voice":
        if (voice === "cloud") setScreen("voice-cloud");
        else setScreen("download");
        return;
      case "voice-cloud":
        if (needsDownload("stt") || needsDownload("tts")) setScreen("download");
        else await finish();
        return;
      case "download":
        if (downloadDone) await finish();
        return;
      case "pet-only":
        await finish();
        return;
      case "done":
        setSaving(true);
        setSaveError("");
        try {
          await invoke("complete_onboarding");
        } catch (error) {
          if (import.meta.env.DEV) {
            window.close();
          } else {
            setSaveError(error instanceof Error ? error.message : String(error));
            setSaving(false);
          }
        }
        return;
    }
  };

  const canContinue = (): boolean => {
    switch (screen) {
      case "welcome": return true;
      case "engine": return true;
      case "engine-agent":
      case "engine-local":
      case "engine-key": return engineContinueOk;
      case "hearing": return true;
      case "hearing-cloud": return true;
      case "voice": return true;
      case "voice-cloud": return true;
      case "download": return downloadDone;
      case "pet-only": return !saving && !saveError;
      case "done": return !saving;
    }
  };

  const continueLabel = (): string => {
    // Bookends use their own CTA copy.
    if (screen === "welcome") return "Let's go";
    if (screen === "done") return saving ? "Saving" : "Meet Mellow";
    if (screen === "pet-only") return saving ? "Saving" : "Meet Mellow";
    return "Continue";
  };

  // ---- Render helpers ----

  const patchLlm = (fields: Partial<Transport>) =>
    setLlm((cur) => (cur ? { ...cur, ...fields } : cur));
  const patchStt = (fields: Partial<NonNullable<typeof stt>>) =>
    setStt((cur) => (cur ? { ...cur, ...fields } : cur));
  const patchTts = (fields: Partial<NonNullable<typeof tts>>) =>
    setTts((cur) => (cur ? { ...cur, ...fields } : cur));

  const chooseProvider = (which: "llm" | "stt" | "tts", provider: string) => {
    if (!config) return;
    const preset = config.presets[which][provider];
    const patch =
      which === "llm" ? patchLlm : which === "stt" ? patchStt : patchTts;
    patch({
      provider,
      base_url: preset?.base_url ?? "",
      api_key: "",
      has_api_key: false,
      ...(preset?.model ? { model: preset.model } : {}),
    });
  };

  if (loadError || !config || !llm || !stt || !tts) {
    return (
      <main className="wizard wizard--bookend">
        <div className="wizard-top" />
        <div className="wizard-col">
          <div className="wizard-mat">
            <div className="wizard-dog wizard-step--thinking" />
          </div>
          <div className="wizard-head">
            <h1>Waking up</h1>
            <p className="wizard-sub">
              {loadError
                ? "Mellow's helper hasn't answered yet. It is probably still starting."
                : "One moment."}
            </p>
          </div>
          <div className="wizard-swap">
            {loadError && (
              <button className="button button--secondary" type="button" onClick={load}>
                Try again
              </button>
            )}
          </div>
          <div className="wizard-go" />
        </div>
      </main>
    );
  }

  const back = backFrom();
  const pose = poseOf(screen, downloadSeconds);
  const step = STEP_OF[screen];

  // Bookends: bigger dog (pet-only keeps the smaller one).
  const bookend = screen === "welcome" || screen === "done";
  // Wide column for engine; form screens pair labels.
  const wide = screen === "engine";
  const localForm = screen === "engine-local";
  const form = !bookend && !wide && screen !== "download";

  return (
    <main className={`wizard${bookend ? " wizard--bookend" : ""}`}>
      <div className="wizard-top">
        {back !== null && !bookend ? (
          <button
            className="wizard-back"
            type="button"
            disabled={saving}
            onClick={() => setScreen(back)}
          >
            {"‹"} back
          </button>
        ) : (
          <span />
        )}
        {!bookend && step !== null && step !== "done" && (
          <div className="wizard-dots" aria-hidden="true">
            {WORKING_STEPS.map((name) => (
              <i
                key={name}
                className={
                  name === step
                    ? "is-now"
                    : WORKING_STEPS.indexOf(name) < WORKING_STEPS.indexOf(step as never)
                      ? "is-done"
                      : ""
                }
              />
            ))}
          </div>
        )}
      </div>

      <div className="wizard-col">
        <div className="wizard-mat">
          <div className={`wizard-dog wizard-step--${pose}`} />
        </div>

        <div key={`${screen}-head`} className="wizard-head">
          <h1>{HEADS[screen].title}</h1>
          <p className="wizard-sub">{HEADS[screen].sub}</p>
        </div>

        <div
          key={screen}
          className={
            "wizard-swap" +
            (wide ? " wizard-swap--wide" : "") +
            (form ? " wizard-swap--form" : "") +
            (localForm ? " wizard-swap--local" : "")
          }
        >
          {renderScreen(config, llm, stt, tts)}
          {saveError && screen !== "pet-only" && <Notice kind="error">{saveError}</Notice>}
        </div>

        <div className="wizard-go">
          <button
            className="button button--primary"
            type="button"
            disabled={!canContinue()}
            onClick={() => void advance()}
          >
            {continueLabel()}
          </button>
        </div>
      </div>
    </main>
  );

  function renderScreen(cfg: ConfigResponse, llmS: Transport, sttS: NonNullable<typeof stt>, ttsS: NonNullable<typeof tts>) {
    switch (screen) {
      case "welcome":
        return (
          <ul className="can">
            {CAN_DO.map(([label, icon]) => (
              <li key={label}>
                <span className="can-label">{label}</span>
                {icon}
              </li>
            ))}
          </ul>
        );

      case "engine":
        return (
          <>
            <fieldset className="choices choices--row">
              {(
                [
                  ["agent", "An agent", agentSelected?.installed
                    ? `${agentSelected.label}, signed in with your plan. No API key.`
                    : "Claude Code or Codex, on your plan. No API key."],
                  ["local", "A local model", "Ollama on this machine. Private and offline."],
                  ["key", "An API key", "Any OpenAI compatible provider. Fastest answers."],
                ] as const
              ).map(([value, label, hint]) => (
                <label className="choice" key={value}>
                  <input
                    type="radio"
                    name="engine"
                    value={value}
                    checked={engine === value}
                    onChange={() => goEngine(value)}
                  />
                  <span className="choice-label">{label}</span>
                  <span className="choice-hint">{hint}</span>
                </label>
              ))}
            </fieldset>
            <fieldset className="choices choices--apart">
              <label className="choice">
                <input
                  type="radio"
                  name="engine"
                  value="pet"
                  checked={engine === "pet"}
                  onChange={() => goEngine("pet")}
                />
                <span className="choice-label">Just the pet</span>
                <span className="choice-hint">
                  No AI at all. Pomodoro, reminders, and a dog. Nothing downloads.
                </span>
              </label>
            </fieldset>
          </>
        );

      case "engine-agent": {
        const notInstalled = variant.current === "notinstalled" || agentSelected?.installed === false;
        return (
          <>
            <AgentFields
              provider={agentProvider}
              model={agentModel}
              speed={agentSpeed}
              agents={
                variant.current === "notinstalled"
                  ? agents.map((a) => (a.id === agentProvider ? { ...a, installed: false } : a))
                  : agents
              }
              busy={connectState.state === "testing" ? "connect" : null}
              onPatch={(fields) => {
                // Changing agent inputs clears a prior successful test.
                setEngineTest({ state: "idle" });
                setConnectState({ state: "idle" });
                if (fields.provider !== undefined) setAgentProvider(fields.provider);
                if (fields.model !== undefined) setAgentModel(fields.model);
                if (fields.agent_speed !== undefined) setAgentSpeed(fields.agent_speed);
              }}
              onConnect={() => void runConnect()}
              notice={null}
            />
            {notInstalled && (
              <Notice kind="error">
                Not installed yet. Run the command above in any terminal, then press Refresh
                on the agent list.
              </Notice>
            )}
            {connectState.state === "ok" && engineTest.state !== "ok" && (
              <Notice kind="ok">{connectState.detail}</Notice>
            )}
            {connectState.state === "failed" && <Notice kind="error">{connectState.detail}</Notice>}
            {engineTest.state === "testing" && (
              <p className="field-note">Asking {agentSelected?.label ?? "the agent"} for a word…</p>
            )}
            <button
              className="button button--secondary"
              type="button"
              disabled={engineTest.state === "testing"}
              onClick={() => void runEngineTest()}
            >
              {engineTest.state === "testing" ? "Testing…" : "Test connection"}
            </button>
            {engineTest.state === "ok" && (
              <Notice kind="ok">
                <span className="notice-status">
                  <svg viewBox="0 0 16 16" aria-hidden="true">
                    <circle cx="8" cy="8" r="6.5" />
                    <path d="m4.8 8.1 2.1 2.1 4.4-4.5" />
                  </svg>
                  Connected — {agentSelected?.label ?? "The agent"} answered successfully.
                </span>
              </Notice>
            )}
            {engineTest.state === "failed" && <Notice kind="error">{engineTest.detail}</Notice>}
          </>
        );
      }

      case "engine-local":
        return (
          <>
            <section className="local-setup-note" aria-labelledby="local-setup-title">
              <h2 id="local-setup-title">Set up Ollama first</h2>
              <ol>
                <li>
                  Install Ollama from{" "}
                  <button
                    className="inline-link"
                    type="button"
                    onClick={() => void openUrl("https://ollama.com/download")}
                  >
                    ollama.com/download
                  </button>
                  , then open it.
                </li>
                <li>
                  In a terminal, run <code>{"ollama run <model-name>"}</code> to download
                  the model you want. For example, <code>ollama run gemma3:4b</code>.
                </li>
                <li>
                  Keep Ollama open, enter that exact model name below, then select{" "}
                  <strong>Test connection</strong>.
                </li>
              </ol>
            </section>
            <Field label="Installed model name" hint="Use the exact name from your command, such as gemma3:4b.">
              <input
                value={llmS.model}
                onChange={(event) => patchLlm({ model: event.target.value })}
                placeholder="gemma3:4b"
                spellCheck={false}
              />
            </Field>
            <div className="history-bar">
              <button
                className="button button--secondary"
                type="button"
                disabled={engineTest.state === "testing"}
                onClick={() => void runEngineTest()}
              >
                {engineTest.state === "testing" ? "Testing…" : "Test connection"}
              </button>
            </div>
            {engineTest.state === "ok" && <Notice kind="ok">Answered: {engineTest.detail}</Notice>}
            {(engineTest.state === "failed" || variant.current === "nofound") && (
              <>
                <Notice kind="error">
                  {ollamaMissing || variant.current === "nofound"
                    ? "Ollama isn't running, or was never installed."
                    : engineTest.detail}
                </Notice>
                {(ollamaMissing || variant.current === "nofound") && (
                  <p className="field-note">
                    Open Ollama, then run <code>ollama run gemma3:4b</code> in a terminal and
                    try again. Or go back and pick an agent, which needs no download at all.
                  </p>
                )}
              </>
            )}
          </>
        );

      case "engine-key":
        return (
          <>
            <CloudFields
              name="llm"
              section={llmS}
              presets={cfg.presets.llm}
              modelHint="Exact model ID from the provider"
              onPatch={patchLlm}
              onProvider={(provider) => chooseProvider("llm", provider)}
            />
            <div className="history-bar">
              <button
                className="button button--secondary"
                type="button"
                disabled={engineTest.state === "testing"}
                onClick={() => void runEngineTest()}
              >
                {engineTest.state === "testing" ? "Testing…" : "Test connection"}
              </button>
            </div>
            {engineTest.state === "ok" && <Notice kind="ok">Answered: {engineTest.detail}</Notice>}
            {(engineTest.state === "failed" || variant.current === "failed") && (
              <Notice kind="error">
                {variant.current === "failed" && engineTest.state !== "failed"
                  ? "The provider said no. Check the key and the model name."
                  : engineTest.detail}
              </Notice>
            )}
          </>
        );

      case "hearing":
        return (
          <>
            <fieldset className="choices">
              {(
                [
                  ["local", "On device", "Parakeet runs here. About 600MB, once."],
                  ["cloud", "In the cloud", "Your voice is recorded and uploaded per question."],
                ] as const
              ).map(([value, label, hint]) => (
                <label className="choice" key={value}>
                  <input
                    type="radio"
                    name="hearing"
                    value={value}
                    checked={hearing === value}
                    onChange={() => setHearing(value)}
                  />
                  <span className="choice-label">{label}</span>
                  <span className="choice-hint">{hint}</span>
                </label>
              ))}
            </fieldset>
            {hearing === "local" && (
              <>
                <Field label="Microphone" hint="Automatic suits almost everyone.">
                  <select
                    value={sttS.input_device ?? ""}
                    onChange={(event) => patchStt({ input_device: event.target.value || null })}
                  >
                    <option value="">Automatic</option>
                    {devices.map((device) => (
                      <option key={device.name} value={device.name}>
                        {device.name}
                        {device.default ? " · default" : ""}
                      </option>
                    ))}
                  </select>
                </Field>
                <div className="history-bar">
                  <button
                    className="button button--secondary"
                    type="button"
                    disabled={sttTest.state === "testing"}
                    onClick={() => void runSttTest()}
                  >
                    {sttTest.state === "testing" ? "Listening for 5 seconds…" : "Test microphone"}
                  </button>
                </div>
                {sttTest.state === "ok" && <Notice kind="ok">{sttTest.detail}</Notice>}
                {sttTest.state === "failed" && <Notice kind="error">{sttTest.detail}</Notice>}
              </>
            )}
          </>
        );

      case "hearing-cloud":
        return (
          <>
            <CloudFields
              name="stt"
              section={sttS}
              presets={cfg.presets.stt}
              modelHint="e.g. whisper-large-v3-turbo"
              onPatch={patchStt}
              onProvider={(provider) => chooseProvider("stt", provider)}
            />
            <div className="history-bar">
              <button
                className="button button--secondary"
                type="button"
                disabled={sttTest.state === "testing"}
                onClick={() => void runSttTest()}
              >
                {sttTest.state === "testing" ? "Listening for 5 seconds…" : "Test microphone"}
              </button>
            </div>
            {sttTest.state === "ok" && <Notice kind="ok">{sttTest.detail}</Notice>}
            {sttTest.state === "failed" && <Notice kind="error">{sttTest.detail}</Notice>}
          </>
        );

      case "download": {
        const needed = (["stt", "tts"] as const).filter((w) => needsDownload(w));
        return (
          <>
            {needed.map((which) => {
              const p = liveProgress[which];
              const pct = p.total > 0 ? Math.min(100, Math.round((p.done / p.total) * 100)) : 0;
              // Hide useless 0-of-0 MB totals.
              const hasModelSize = p.total >= 500_000;
              const label = which === "stt" ? "Ears, Parakeet" : "Voice, Kokoro";
              return (
                <div className="progress" key={which}>
                  <div className="progress-head">
                    <span>{label}</span>
                    <span className="progress-num">
                      {p.state === "failed"
                        ? "failed"
                        : p.state === "done"
                          ? "available"
                          : `${pct}%`}
                    </span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{
                        "--progress-scale": (p.state === "done" ? 100 : pct) / 100,
                      } as CSSProperties}
                    />
                  </div>
                  {/* Hide size until the first tick reports a total. */}
                  {hasModelSize && (
                    <span className="progress-num">
                      {mb(p.done)} of {mb(p.total)} MB
                    </span>
                  )}
                  {p.state === "done" && !hasModelSize && (
                    <span className="progress-num">Already on this device</span>
                  )}
                </div>
              );
            })}
            {downloadFailed && (
              <>
                <Notice kind="error">
                  {liveProgress[downloadFailed].error ||
                    `The ${downloadFailed === "stt" ? "ears" : "voice"} download failed partway.`}
                </Notice>
                <div className="history-bar">
                  <button
                    className="button button--secondary"
                    type="button"
                    onClick={() => {
                      setProgress((cur) => ({ ...cur, [downloadFailed]: IDLE_PROGRESS }));
                      void startDownloads();
                    }}
                  >
                    Try again
                  </button>
                  <button
                    className="button button--secondary"
                    type="button"
                    onClick={() => {
                      if (downloadFailed === "stt") {
                        setHearing("cloud");
                        setScreen("hearing-cloud");
                      } else {
                        setVoice("cloud");
                        setScreen("voice-cloud");
                      }
                    }}
                  >
                    Use the cloud instead
                  </button>
                </div>
              </>
            )}
            {downloadRunning && !downloadFailed && nearly && (
              <p className="field-note">Nearly there.</p>
            )}
            {voice === "local" && ttsAvailable && (
              <div className="download-voice-test">
                <p className="field-note">Your selected voice is ready. Hear it before continuing.</p>
                {testVoice()}
              </div>
            )}
          </>
        );
      }

      case "voice":
        return (
          <>
            <fieldset className="choices">
              {(
                [
                  ["local", "On device", "Kokoro runs here. About 340MB, once."],
                  ["cloud", "In the cloud", "Hosted voices. A fetch per sentence."],
                ] as const
              ).map(([value, label, hint]) => (
                <label className="choice" key={value}>
                  <input
                    type="radio"
                    name="voice"
                    value={value}
                    checked={voice === value}
                    onChange={() => {
                      setVoice(value);
                      setTtsTest({ state: "idle" });
                    }}
                  />
                  <span className="choice-label">{label}</span>
                  <span className="choice-hint">{hint}</span>
                </label>
              ))}
            </fieldset>
            {voice === "local" ? (
              <>
                <Field label="Voice">
                  <select
                    value={ttsS.local_voice}
                    onChange={(event) => {
                      patchTts({ local_voice: event.target.value });
                      setTtsTest({ state: "idle" });
                    }}
                  >
                    {Object.entries(cfg.tts_voices).map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                </Field>
                {testVoice()}
              </>
            ) : (
              <p className="field-note">The voice is picked on the next screen.</p>
            )}
          </>
        );

      case "voice-cloud":
        return (
          <>
            <CloudFields
              name="tts"
              section={ttsS}
              presets={cfg.presets.tts}
              modelHint="Exact model ID from the provider"
              onPatch={(fields) => {
                patchTts(fields);
                setTtsTest({ state: "idle" });
              }}
              onProvider={(provider) => {
                chooseProvider("tts", provider);
                setTtsTest({ state: "idle" });
              }}
            />
            <Field label="Voice" optional hint="Blank if the model has only one voice.">
              <input
                value={ttsS.voice}
                onChange={(event) => {
                  patchTts({ voice: event.target.value });
                  setTtsTest({ state: "idle" });
                }}
                placeholder="Voice ID, if the provider asks for one"
                spellCheck={false}
              />
            </Field>
            {testVoice()}
          </>
        );

      case "pet-only":
        return (
          <>
            <div className="wizard-cols">
              <ul className="wizard-plain">
                <b>Still works</b>
                <li>Petting and dragging him around</li>
                <li>Pomodoro timer</li>
                <li>Reminders</li>
                <li>Stay quiet, hide, tray</li>
              </ul>
              <ul className="wizard-plain">
                <b>Off</b>
                <li>Answering questions</li>
                <li>Speech</li>
                <li>Screen reading</li>
                <li>Any downloads</li>
              </ul>
            </div>
            <p className="field-note">
              Changed your mind? Settings can turn his brain on later.
            </p>
            {saveError && <Notice kind="error">{saveError}</Notice>}
          </>
        );

      case "done":
        return (
          <>
            <div className="keycaps">
              <span className="keycap">Ctrl</span>
              <span className="keycaps-plus">+</span>
              <span className="keycap">Shift</span>
              <span className="keycaps-plus">+</span>
              <span className="keycap">Space</span>
            </div>
            <p className="field-note">
              Right click him for everything else. Settings lives in his menu and in the tray.
            </p>
            <p className="field-note motion-note">
              <strong>For Mellow&rsquo;s full animations:</strong> open Windows Settings, then
              Accessibility, Visual effects, and turn on Animation effects.
            </p>
          </>
        );
    }
  }

  function testVoice() {
    const waitsForDownload = voice === "local" && !ttsAvailable;
    return (
      <>
        <div className="history-bar">
          <button
            className="button button--secondary"
            type="button"
            disabled={ttsTest.state === "testing" || waitsForDownload}
            onClick={() => void runTtsTest()}
          >
            {ttsTest.state === "testing"
              ? "Speaking…"
              : waitsForDownload
                ? "Test after download"
                : "Test voice"}
          </button>
        </div>
        {waitsForDownload && (
          <p className="field-note">
            You can test this voice after the one-time download on the next step.
          </p>
        )}
        {ttsTest.state === "ok" && <Notice kind="ok">{ttsTest.detail}</Notice>}
        {ttsTest.state === "failed" && <Notice kind="error">{ttsTest.detail}</Notice>}
      </>
    );
  }
}

function mb(bytes: number): string {
  return (bytes / 1e6).toFixed(0);
}
