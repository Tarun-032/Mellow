import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import {
  Action,
  AgentFields,
  Capability,
  CloudFields,
  Mode,
  ModeToggle,
  Notice,
  NoticeData,
  Preset,
  request,
  Transport,
  type AgentInfo,
} from "../ui/fields";
import "./settings.css";

type SettingsData = {
  llm: Transport & {
    max_tokens: number;
    reasoning_effort: string;
    agent_speed: "fast" | "balanced" | "deep";
    temperature: number;
    /** auto guesses from the model name; on/off override the guess. */
    vision: string;
  };
  stt: Transport & {
    /** Device name, or null for the Windows default (not an index). */
    local_model: string;
    input_device: string | null;
  };
  tts: Transport & {
    local_voice: string;
    voice: string;
    speech_speed: number;
    speak: boolean;
  };
  system_prompt: string;
  /** Session event log master switch. */
  remember_conversations: boolean;
  /** AI enabled; false = pet-only mode. */
  ai_enabled: boolean;
};

type Device = {
  name: string;
  channels: number;
  default_samplerate: number;
  default: boolean;
};

type ConfigResponse = {
  settings: SettingsData;
  presets: Record<Capability, Record<string, Preset>>;
  stt_models: Record<string, string>;
  tts_voices: Record<string, string>;
  reasoning_efforts: string[];
  vision_modes: string[];
  default_prompt: string;
};

type SaveResponse = {
  settings: SettingsData;
  engine_changed: boolean;
};

type EngineSaveAction = "save" | "connect";

function engineBaseUrl(value: string): string {
  let normalized = value.trim().replace(/\/+$/, "");
  for (const endpoint of [
    "/chat/completions",
    "/audio/speech",
    "/audio/transcriptions",
  ]) {
    if (normalized.endsWith(endpoint)) {
      normalized = normalized.slice(0, -endpoint.length);
      break;
    }
  }
  return normalized;
}

/** Fields that define the conversation engine (see _engine_signature). */
function engineKey(settings: SettingsData): string {
  if (!settings.ai_enabled) return JSON.stringify(["pet"]);
  const { mode, provider, base_url, model } = settings.llm;
  if (mode === "agent") {
    return JSON.stringify(["ai", mode, provider.trim(), model.trim()]);
  }
  return JSON.stringify([
    "ai",
    mode,
    provider.trim(),
    engineBaseUrl(base_url),
    model.trim(),
  ]);
}

type Voice = { id: string; name: string };

type SettingsPage =
  | "customize"
  | "engine"
  | "stt"
  | "tts"
  | "sessions"
  | "advanced";

const SETTINGS_PAGES: Array<{
  id: SettingsPage;
  label: string;
  description: string;
}> = [
  {
    id: "customize",
    label: "Customize Mellow",
    description: "Appearance and personality",
  },
  {
    id: "engine",
    label: "Engine",
    description: "Choose how Mellow thinks",
  },
  {
    id: "stt",
    label: "Speech to text",
    description: "Microphone and transcription",
  },
  {
    id: "tts",
    label: "Text to speech",
    description: "Voice and playback",
  },
  {
    id: "sessions",
    label: "Sessions",
    description: "Saved conversations",
  },
  {
    id: "advanced",
    label: "Advanced",
    description: "Model behavior and vision",
  },
];

/** One line of the history index — no session file has been opened yet. */
type SessionEntry = {
  id: string;
  /** "conversation" today; "agent" once background agents run (roadmap 16). */
  kind: string;
  /** The session that spawned this one, for an agent run. Empty otherwise. */
  parent: string;
  started_at: string;
  ended_at: string;
  title: string;
  /** Exchange count (not raw event count). */
  turns: number;
  events: number;
};

/** One session event (typed or generic). */
type SessionEvent = {
  seq: number;
  ts: string;
  type: string;
  text?: string;
  model?: string;
  provider?: string;
  aborted?: boolean;
};

/** Session time range for the history list. */
function span(started: string, ended: string): string {
  const from = new Date(started);
  const to = new Date(ended);
  if (Number.isNaN(from.valueOf())) return "Unknown time";
  const clock = (d: Date) =>
    d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  const date = (d: Date) =>
    d.toDateString() === new Date().toDateString()
      ? "Today"
      : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  if (Number.isNaN(to.valueOf())) return `${date(from)}, ${clock(from)}`;
  // Include both dates when the range crosses midnight.
  return from.toDateString() === to.toDateString()
    ? `${date(from)}, ${clock(from)} – ${clock(to)}`
    : `${date(from)} ${clock(from)} – ${date(to)} ${clock(to)}`;
}

/** Which button is running. Only one action at a time makes sense. */
type Notice = NoticeData;

/** What to type in the model box, per hosted voice provider. */
const TTS_MODEL_HINTS: Record<string, string> = {
  elevenlabs: "e.g. eleven_flash_v2_5 or eleven_multilingual_v2",
  openrouter: "e.g. fish-audio/s2.1-pro-free:free",
  openai: "e.g. gpt-4o-mini-tts",
  groq: "e.g. canopylabs/orpheus-v1-english",
};

const EFFORT_LABELS: Record<string, string> = {
  "": "Provider default",
  none: "None · no thinking",
  default: "Default · Qwen thinking on",
  low: "Low",
  medium: "Medium",
  high: "High",
};

const VISION_LABELS: Record<string, string> = {
  auto: "Auto · guess from the model name",
  on: "Can see screenshots",
  off: "Cannot see screenshots",
};

export default function Settings() {
  const [activePage, setActivePage] = useState<SettingsPage>("engine");
  const [navQuery, setNavQuery] = useState("");
  const [form, setForm] = useState<SettingsData | null>(null);
  const [savedEngineKey, setSavedEngineKey] = useState<string | null>(null);
  const [pendingEngineAction, setPendingEngineAction] =
    useState<EngineSaveAction | null>(null);
  const [presets, setPresets] = useState<Record<Capability, Record<string, Preset>> | null>(null);
  const [sttModels, setSttModels] = useState<Record<string, string>>({});
  const [ttsVoices, setTtsVoices] = useState<Record<string, string>>({});
  const [efforts, setEfforts] = useState<string[]>([""]);
  const [visionModes, setVisionModes] = useState<string[]>(["auto"]);
  const [defaultPrompt, setDefaultPrompt] = useState("");
  const [devices, setDevices] = useState<Device[]>([]);
  // Live agent CLI + model catalogue (refresh re-probes).
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const loadAgents = (refresh = false) => {
    request<{ agents: AgentInfo[] }>(`/agents${refresh ? "?refresh=true" : ""}`)
      .then((result) => setAgents(result.agents))
      .catch(() => void 0);
  };
  // Cloud model list for the key currently in the box.
  const [voices, setVoices] = useState<Voice[]>([]);
  const [busy, setBusy] = useState<Action | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  // History: index always; transcript on open.
  const [history, setHistory] = useState<SessionEntry[] | null>(null);
  const [openSession, setOpenSession] = useState<{
    entry: SessionEntry;
    events: SessionEvent[];
  } | null>(null);
  const [historyBusy, setHistoryBusy] = useState(false);

  const loadHistory = () => {
    setHistoryBusy(true);
    request<{ sessions: SessionEntry[] }>("/history")
      .then((result) => setHistory(result.sessions))
      .catch(() => setHistory([]))
      .finally(() => setHistoryBusy(false));
  };

  const openTranscript = (entry: SessionEntry) => {
    if (openSession?.entry.id === entry.id) {
      setOpenSession(null);
      return;
    }
    setHistoryBusy(true);
    request<{ events: SessionEvent[] }>(`/history/${entry.id}`)
      .then((result) => setOpenSession({ entry, events: result.events }))
      .catch(() => void 0)
      .finally(() => setHistoryBusy(false));
  };

  const clearHistory = async () => {
    if (!window.confirm("Delete every saved conversation? This cannot be undone.")) return;
    setHistoryBusy(true);
    try {
      await request("/history/clear", { method: "POST" });
      setOpenSession(null);
      setHistory([]);
    } catch {
      void 0;
    } finally {
      setHistoryBusy(false);
    }
  };

  useEffect(() => {
    Promise.all([
      request<ConfigResponse>("/config"),
      request<{ devices: Device[] }>("/audio/devices"),
    ])
      .then(([config, audio]) => {
        setForm(config.settings);
        setSavedEngineKey(engineKey(config.settings));
        setPresets(config.presets);
        setSttModels(config.stt_models);
        setTtsVoices(config.tts_voices);
        setEfforts(config.reasoning_efforts);
        setVisionModes(config.vision_modes);
        setDefaultPrompt(config.default_prompt);
        setDevices(audio.devices);
      })
      .catch((error) =>
        setNotice({
          where: "save",
          kind: "error",
          text: `Mellow sidecar is unavailable: ${error.message}`,
        }),
      );
    loadHistory();
    loadAgents();
  }, []);

  // Refetch history when the window regains focus.
  useEffect(() => {
    window.addEventListener("focus", loadHistory);
    return () => window.removeEventListener("focus", loadHistory);
  }, []);

  /** Patch one capability's fields. Every field in the form goes through here. */
  const patch = (name: Capability, fields: Partial<SettingsData[Capability]>) => {
    setForm((current) =>
      current ? { ...current, [name]: { ...current[name], ...fields } } : current,
    );
    setNotice(null);
  };

  /** Provider switch: set base URL and clear saved-key placeholder. */
  const chooseProvider = (name: Capability, provider: string) => {
    const preset = presets?.[name][provider];
    if (name === "tts") setVoices([]);
    patch(name, {
      provider,
      base_url: preset?.base_url ?? "",
      api_key: "",
      has_api_key: false,
      ...(preset?.model ? { model: preset.model } : {}),
    });
  };

  const run = async (action: Action, formOverride?: SettingsData) => {
    const current = formOverride ?? form;
    if (!current) return;
    setBusy(action);
    setNotice(null);
    const ok = (text: string) => setNotice({ where: action, kind: "ok", text });
    try {
      if (action === "save") {
        const result = await request<SaveResponse>("/config", {
          method: "PUT",
          body: JSON.stringify(current),
        });
        setForm(result.settings);
        setSavedEngineKey(engineKey(result.settings));
        if (result.engine_changed) {
          setOpenSession(null);
          loadHistory();
          ok("Engine changed. Your next message starts a new session.");
        } else {
          ok("Settings saved.");
        }
      } else if (action === "llm") {
        const result = await request<{ reply: string }>("/config/test", {
          method: "POST",
          body: JSON.stringify(current),
        });
        ok(`Connected successfully: ${result.reply}`);
      } else if (action === "connect") {
        // Probe first; only open sign-in console if needed.
        const result = await request<{
          ok: boolean;
          signed_in: boolean;
          model_ok: boolean;
          vision_ok: boolean;
          detail: string;
        }>(
          "/agents/login",
          {
            method: "POST",
            body: JSON.stringify({
              agent: current.llm.provider,
              model: current.llm.model,
              agent_speed: current.llm.agent_speed,
            }),
          },
        );
        if (result.signed_in && result.ok && result.model_ok && result.vision_ok) {
          // Already signed in: save agent and refresh models.
          const saved = await request<SaveResponse>("/config", {
            method: "PUT",
            body: JSON.stringify(current),
          });
          setForm(saved.settings);
          setSavedEngineKey(engineKey(saved.settings));
          if (saved.engine_changed) {
            setOpenSession(null);
            loadHistory();
          }
          loadAgents(true);
          ok(
            `${
              saved.engine_changed
                ? "Signed in and started a fresh session."
                : "Signed in and saved."
            } ${result.detail}`,
          );
        } else if (result.signed_in) {
          throw new Error(result.detail || "The selected model failed vision verification.");
        } else {
          ok(
            `${
              result.detail ? `${result.detail} — ` : ""
            }a sign-in window opened. Sign in there, close it, then press Connect again.`,
          );
        }
      } else if (action === "stt") {
        const result = await request<{
          transcript: string;
          peak: number;
          rms: number;
          model: string;
          backend: string;
          device: string;
        }>("/stt/test", { method: "POST", body: JSON.stringify(form) });
        // Include device name with the level meter.
        ok(
          result.transcript
            ? `Heard “${result.transcript}” on ${result.device} · peak ${result.peak.toFixed(3)}`
            : `No speech reached ${result.device} · peak ${result.peak.toFixed(3)}, RMS ${result.rms.toFixed(3)}`,
        );
      } else if (action === "voices") {
        const result = await request<{ voices: Voice[] }>("/tts/voices", {
          method: "POST",
          body: JSON.stringify(form),
        });
        setVoices(result.voices);
        // Default the dropdown to the first device.
        if (!current.tts.voice && result.voices.length) {
          patch("tts", { voice: result.voices[0].id });
        }
        ok(
          result.voices.length
            ? `Found ${result.voices.length} voices on your account.`
            : "That key works, but the account has no voices yet.",
        );
      } else {
        const result = await request<{ backend: string; voice: string; seconds: number }>(
          "/tts/test",
          { method: "POST", body: JSON.stringify(form) },
        );
        ok(`Spoke ${result.seconds}s with ${result.backend}`);
      }
    } catch (error) {
      setNotice({
        where: action,
        kind: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(null);
    }
  };

  const requestEngineAction = (action: EngineSaveAction) => {
    if (form && savedEngineKey !== null && engineKey(form) !== savedEngineKey) {
      setPendingEngineAction(action);
      return;
    }
    void run(action);
  };

  const confirmEngineChange = () => {
    const action = pendingEngineAction;
    setPendingEngineAction(null);
    if (action) void run(action);
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    requestEngineAction("save");
  };

  if (!form || !presets) {
    return (
      <main className="settings-shell">
        <p className={`notice ${notice?.kind === "error" ? "notice--error" : ""}`}>
          {notice?.text ?? "Loading Mellow’s settings…"}
        </p>
      </main>
    );
  }

  /** Keep mode and provider aligned (LLM local preset only). */
  const chooseMode = (name: Capability, mode: Mode | "pet") => {
    // Pet-only is the master AI switch, not a transport.
    if (name === "llm" && mode === "pet") {
      setForm((current) =>
        current ? { ...current, ai_enabled: false } : current,
      );
      setNotice(null);
      return;
    }
    if (name === "llm" && !form.ai_enabled) {
      setForm((current) => (current ? { ...current, ai_enabled: true } : current));
    }
    // Agent mode keeps credentials; only ensures a real agent provider.
    if (name === "llm" && mode === "agent") {
      if (!agents.some((item) => item.id === form.llm.provider)) {
        const first = agents.find((item) => item.installed) ?? agents[0];
        if (first) patch("llm", { provider: first.id });
      }
      patch(name, { mode });
      return;
    }
    const wantLocal = mode === "local";
    if (Boolean(presets[name][form[name].provider]?.local) !== wantLocal) {
      const match = Object.entries(presets[name]).find(
        ([, item]) => Boolean(item.local) === wantLocal,
      );
      if (match) chooseProvider(name, match[0]);
    }
    // Remaining cases are real transport modes.
    patch(name, { mode: mode as Mode });
  };

  const testButton = (action: Action, idle: string, running: string) => (
    <button
      className="button button--secondary"
      type="button"
      disabled={busy !== null}
      onClick={() => void run(action)}
    >
      {busy === action ? running : idle}
    </button>
  );

  const sectionNotice = (where: Action): ReactNode =>
    notice && notice.where === where ? (
      <Notice kind={notice.kind}>{notice.text}</Notice>
    ) : null;

  const currentPage =
    SETTINGS_PAGES.find((page) => page.id === activePage) ?? SETTINGS_PAGES[1];
  const visiblePages = SETTINGS_PAGES.filter((page) =>
    `${page.label} ${page.description}`.toLowerCase().includes(navQuery.trim().toLowerCase()),
  );

  return (
    <main className="settings-shell">
      <aside className="settings-sidebar" aria-label="Settings navigation">
        <div className="settings-brand">
          <span className="settings-brand__pet" aria-hidden="true" />
          <span>
            <strong>Mellow</strong>
            <small>Settings</small>
          </span>
        </div>

        <label className="settings-search">
          <span className="sr-only">Search settings</span>
          <input
            type="search"
            value={navQuery}
            onChange={(event) => setNavQuery(event.target.value)}
            placeholder="Search settings"
          />
        </label>

        <nav className="settings-nav">
          <p className="settings-nav__label">Settings</p>
          {visiblePages.map((page) => (
            <button
              key={page.id}
              type="button"
              className="settings-nav__item"
              aria-current={activePage === page.id ? "page" : undefined}
              onClick={() => setActivePage(page.id)}
            >
              <span>{page.label}</span>
              <small>{page.description}</small>
            </button>
          ))}
          {visiblePages.length === 0 && (
            <p className="settings-nav__empty">No matching settings</p>
          )}
        </nav>

      </aside>

      <form className="settings-workspace" onSubmit={submit}>
        <header className="settings-topbar">
          <div>
            <h1>{currentPage.label}</h1>
            <p>{currentPage.description}</p>
          </div>
        </header>

        <div className="settings-content">
          {activePage === "customize" && (
            <section className="settings-page" aria-labelledby="customize-heading">
              <div className="page-heading">
                <h2 id="customize-heading">Make Mellow yours</h2>
                <p>Appearance, behavior, and pet controls will live together here.</p>
              </div>
              <div className="customize-empty">
                <span className="customize-empty__mark" aria-hidden="true">M</span>
                <div>
                  <h3>Coming in a future update</h3>
                  <p>Appearance, behavior, and more ways to personalize Mellow are coming soon.</p>
                </div>
              </div>
            </section>
          )}

          {activePage === "engine" && (
            <section className="settings-page" aria-labelledby="engine-heading">
              <div className="page-heading">
                <h2 id="engine-heading">Choose how Mellow thinks</h2>
                <p>Pick one engine. Only the setup for that choice appears below.</p>
              </div>
              <div className="settings-group settings-group--modes">
                <ModeToggle
                  name="llm"
                  value={!form.ai_enabled ? "pet" : form.llm.mode}
                  entries={[
                    ["local", "On device", "Ollama on this machine"],
                    ["cloud", "Cloud", "Fastest answers · use an API key"],
                    ["agent", "Agent", "Use Claude Code or Codex"],
                    ["pet", "Just the pet", "No model or microphone"],
                  ]}
                  onChange={(mode) => chooseMode("llm", mode as Mode | "pet")}
                />
              </div>
              <div className="settings-group">
                {!form.ai_enabled ? (
                  <div className="empty-state">
                    <h3>No engine selected</h3>
                    <p>Mellow stays as a desktop pet. No model runs and the microphone stays off.</p>
                  </div>
                ) : form.llm.mode === "cloud" ? (
                  <CloudFields
                    name="llm"
                    section={form.llm}
                    presets={presets.llm}
                    modelHint="Exact model ID from the provider"
                    onPatch={(fields) => patch("llm", fields)}
                    onProvider={(provider) => chooseProvider("llm", provider)}
                  />
                ) : form.llm.mode === "agent" ? (
                  <AgentFields
                    provider={form.llm.provider}
                    model={form.llm.model}
                    speed={form.llm.agent_speed}
                    agents={agents}
                    busy={busy}
                    onPatch={(fields) => patch("llm", fields)}
                    onConnect={() => requestEngineAction("connect")}
                    notice={sectionNotice("connect")}
                  />
                ) : (
                  <label>
                    Model name
                    <input
                      required
                      value={form.llm.model}
                      onChange={(event) => patch("llm", { model: event.target.value })}
                      placeholder="A model you have pulled in Ollama"
                      spellCheck={false}
                    />
                  </label>
                )}
                {form.ai_enabled && testButton("llm", "Test connection", "Testing...")}
                {form.ai_enabled && sectionNotice("llm")}
              </div>
            </section>
          )}

          {activePage === "stt" && (
            <section className="settings-page" aria-labelledby="stt-heading-new">
              <div className="page-heading">
                <h2 id="stt-heading-new">Speech to text</h2>
                <p>Choose where speech is transcribed and which microphone Mellow hears.</p>
              </div>
              <div className="settings-group settings-group--modes">
                <ModeToggle
                  name="stt"
                  value={form.stt.mode}
                  entries={[
                    ["local", "On device", "Private and available offline"],
                    ["cloud", "Cloud", "Use a hosted transcription model"],
                  ]}
                  onChange={(mode) => chooseMode("stt", mode as Mode)}
                />
              </div>
              <div className="settings-group">
                {form.stt.mode === "cloud" ? (
                  <>
                    <CloudFields
                      name="stt"
                      section={form.stt}
                      presets={presets.stt}
                      modelHint="e.g. whisper-large-v3-turbo"
                      onPatch={(fields) => patch("stt", fields)}
                      onProvider={(provider) => chooseProvider("stt", provider)}
                    />
                    <p className="field-note">
                      Cloud transcription sends each recording to the selected provider.
                    </p>
                  </>
                ) : (
                  <label>
                    Model
                    <select
                      value={form.stt.local_model}
                      onChange={(event) => patch("stt", { local_model: event.target.value })}
                    >
                      {Object.entries(sttModels).map(([value, label]) => (
                        <option key={value} value={value}>{label} - {value}</option>
                      ))}
                    </select>
                  </label>
                )}
                <label>
                  Microphone
                  <select
                    value={form.stt.input_device ?? ""}
                    onChange={(event) =>
                      patch("stt", { input_device: event.target.value || null })
                    }
                  >
                    <option value="">Automatic</option>
                    {devices.map((device) => (
                      <option key={device.name} value={device.name}>
                        {device.name}{device.default ? " - default" : ""}
                      </option>
                    ))}
                  </select>
                </label>
                {testButton("stt", "Test microphone", "Listening for 5 seconds...")}
                {sectionNotice("stt")}
              </div>
            </section>
          )}

          {activePage === "tts" && (
            <section className="settings-page" aria-labelledby="tts-heading-new">
              <div className="page-heading">
                <h2 id="tts-heading-new">Text to speech</h2>
                <p>Select Mellow's voice, where it runs, and how quickly it speaks.</p>
              </div>
              <div className="settings-group settings-group--modes">
                <ModeToggle
                  name="tts"
                  value={form.tts.mode}
                  entries={[
                    ["local", "On device", "Kokoro, offline and instant"],
                    ["cloud", "Cloud", "Use a hosted voice"],
                  ]}
                  onChange={(mode) => chooseMode("tts", mode as Mode)}
                />
              </div>
              <div className="settings-group">
                {form.tts.mode === "cloud" ? (
                  <>
                    <CloudFields
                      name="tts"
                      section={form.tts}
                      presets={presets.tts}
                      modelHint={TTS_MODEL_HINTS[form.tts.provider] ?? "Exact model ID from the provider"}
                      onPatch={(fields) => patch("tts", fields)}
                      onProvider={(provider) => chooseProvider("tts", provider)}
                    />
                    {form.tts.provider === "elevenlabs" ? (
                      <>
                        <label>
                          Voice
                          {voices.length ? (
                            <select
                              value={form.tts.voice}
                              onChange={(event) => patch("tts", { voice: event.target.value })}
                            >
                              {form.tts.voice && !voices.some((item) => item.id === form.tts.voice) && (
                                <option value={form.tts.voice}>{form.tts.voice} - saved</option>
                              )}
                              {voices.map((item) => (
                                <option key={item.id} value={item.id}>{item.name}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              required
                              value={form.tts.voice}
                              onChange={(event) => patch("tts", { voice: event.target.value })}
                              placeholder="Voice ID, or load voices"
                              spellCheck={false}
                            />
                          )}
                        </label>
                        {testButton("voices", "Load voices", "Loading...")}
                        {sectionNotice("voices")}
                      </>
                    ) : (
                      <label>
                        Voice <span className="optional">optional</span>
                        <input
                          value={form.tts.voice}
                          onChange={(event) => patch("tts", { voice: event.target.value })}
                          placeholder="Leave blank if the model has one voice"
                          spellCheck={false}
                        />
                      </label>
                    )}
                    <p className="field-note">
                      Hosted voices are fetched one sentence at a time and may start later.
                    </p>
                  </>
                ) : (
                  <>
                    <label>
                      Model name
                      <input value="Kokoro v1.0 (kokoro-v1.0.onnx)" readOnly />
                    </label>
                    <label>
                      Voice
                      <select
                        value={form.tts.local_voice}
                        onChange={(event) => patch("tts", { local_voice: event.target.value })}
                      >
                        {Object.entries(ttsVoices).map(([value, label]) => (
                          <option key={value} value={value}>{label}</option>
                        ))}
                      </select>
                    </label>
                  </>
                )}
                <label>
                  Speed <span className="field-value">{form.tts.speech_speed.toFixed(2)}x</span>
                  <input
                    type="range"
                    min={0.5}
                    max={2}
                    step={0.05}
                    value={form.tts.speech_speed}
                    onChange={(event) =>
                      patch("tts", { speech_speed: Number(event.target.value) })
                    }
                  />
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={form.tts.speak}
                    onChange={(event) => patch("tts", { speak: event.target.checked })}
                  />
                  Speak answers out loud
                </label>
                {testButton("tts", "Test voice", "Speaking...")}
                {sectionNotice("tts")}
              </div>
            </section>
          )}

          {activePage === "sessions" && (
            <section className="settings-page settings-page--sessions" aria-labelledby="sessions-heading">
              <div className="page-heading page-heading--with-action">
                <div>
                  <h2 id="sessions-heading">Saved sessions</h2>
                  <p>Open any conversation to read the full chat.</p>
                </div>
                <label className="checkbox checkbox--compact">
                  <input
                    type="checkbox"
                    checked={form.remember_conversations}
                    onChange={(event) =>
                      setForm((current) =>
                        current
                          ? { ...current, remember_conversations: event.target.checked }
                          : current,
                      )
                    }
                  />
                  Remember sessions
                </label>
              </div>
              {!form.remember_conversations ? (
                <div className="empty-state">
                  <h3>Session history is off</h3>
                  <p>Turn on Remember sessions to keep conversations on this computer.</p>
                </div>
              ) : (
                <div className="sessions-panel">
                  <div className="history-bar">
                    <span>{history?.length ?? 0} saved</span>
                    <div>
                      <button className="button button--quiet" type="button" disabled={historyBusy} onClick={loadHistory}>
                        Refresh
                      </button>
                      <button
                        className="button button--quiet button--danger"
                        type="button"
                        disabled={historyBusy || !history?.length}
                        onClick={() => void clearHistory()}
                      >
                        Clear all
                      </button>
                    </div>
                  </div>
                  {history === null ? (
                    <div className="empty-state"><p>Loading sessions...</p></div>
                  ) : history.length === 0 ? (
                    <div className="empty-state">
                      <h3>No saved sessions yet</h3>
                      <p>Your conversations will appear here after you talk to Mellow.</p>
                    </div>
                  ) : (
                    <ul className="history-list">
                      {history.map((entry) => (
                        <li key={entry.id}>
                          <button
                            className="history-item"
                            type="button"
                            aria-expanded={openSession?.entry.id === entry.id}
                            onClick={() => openTranscript(entry)}
                          >
                            <span className="history-copy">
                              <strong>{entry.title || "Untitled conversation"}</strong>
                              <small>{span(entry.started_at, entry.ended_at)}</small>
                            </span>
                            <span className="history-count">
                              {entry.turns} exchange{entry.turns === 1 ? "" : "s"}
                            </span>
                          </button>
                          {openSession?.entry.id === entry.id && (
                            <div className="transcript">
                              {openSession.events
                                .filter((event) =>
                                  event.type !== "session_start" && event.type !== "session_ended"
                                )
                                .map((event) => (
                                  <p key={event.seq} className={`line line--${event.type}`}>
                                    <span className="line-who">
                                      {event.type === "user_said" ? "You" : "Mellow"}
                                      {event.aborted ? " (cut off)" : ""}
                                    </span>
                                    {event.text || `[${event.type}]`}
                                  </p>
                                ))}
                            </div>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </section>
          )}

          {activePage === "advanced" && (
            <section className="settings-page" aria-labelledby="advanced-heading">
              <div className="page-heading">
                <h2 id="advanced-heading">Advanced model controls</h2>
                <p>These defaults work well for most people. Change them only when needed.</p>
              </div>
              {!form.ai_enabled ? (
                <div className="empty-state">
                  <h3>Advanced controls are unavailable</h3>
                  <p>Choose an engine first, then return here to tune its behavior.</p>
                </div>
              ) : (
                <>
                  {form.llm.mode !== "agent" && (
                    <div className="settings-group settings-grid">
                      <label>
                        Reasoning effort
                        <select value={form.llm.reasoning_effort} onChange={(event) => patch("llm", { reasoning_effort: event.target.value })}>
                          {efforts.map((value) => (
                            <option key={value} value={value}>{EFFORT_LABELS[value] ?? value}</option>
                          ))}
                        </select>
                        <span className="field-note">Lower values answer sooner.</span>
                      </label>
                      <label>
                        Reply limit
                        <input
                          type="number"
                          min={1}
                          max={8192}
                          value={form.llm.max_tokens}
                          onChange={(event) => patch("llm", { max_tokens: Number(event.target.value) })}
                        />
                        <span className="field-note">Maximum tokens in one response.</span>
                      </label>
                      <label>
                        Temperature
                        <input
                          type="number"
                          min={0}
                          max={2}
                          step={0.1}
                          value={form.llm.temperature}
                          onChange={(event) => patch("llm", { temperature: Number(event.target.value) })}
                        />
                        <span className="field-note">Higher values make answers less predictable.</span>
                      </label>
                    </div>
                  )}
                  <div className="settings-group">
                    <label>
                      Vision
                      <select value={form.llm.vision} onChange={(event) => patch("llm", { vision: event.target.value })}>
                        {visionModes.map((value) => (
                          <option key={value} value={value}>{VISION_LABELS[value] ?? value}</option>
                        ))}
                      </select>
                    </label>
                    <p className="field-note">
                      Controls whether Mellow can inspect screenshots when you ask about the screen.
                    </p>
                    <label>
                      Additional personality instructions <span className="optional">optional</span>
                      <textarea
                        className="prompt-box"
                        value={form.system_prompt}
                        onChange={(event) =>
                          setForm((current) =>
                            current ? { ...current, system_prompt: event.target.value } : current,
                          )
                        }
                        spellCheck={false}
                        placeholder="Add preferences that Mellow should remember in every answer"
                      />
                    </label>
                    <button
                      type="button"
                      className="button button--quiet prompt-reset"
                      disabled={!form.system_prompt}
                      onClick={() =>
                        setForm((current) =>
                          current ? { ...current, system_prompt: defaultPrompt } : current,
                        )
                      }
                    >
                      Clear instructions
                    </button>
                  </div>
                </>
              )}
            </section>
          )}
        </div>

        <footer className="settings-savebar">
          <div>
            {sectionNotice("save")}
          </div>
          <button className="button button--primary" type="submit" disabled={busy !== null}>
            {busy === "save" ? "Saving..." : "Save changes"}
          </button>
        </footer>
      </form>
      {pendingEngineAction && (
        <div
          className="engine-change-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setPendingEngineAction(null);
          }}
        >
          <section
            className="engine-change-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="engine-change-title"
            aria-describedby="engine-change-description"
          >
            <p className="engine-change-dialog__eyebrow">New conversation</p>
            <h2 id="engine-change-title">Change Mellow's engine?</h2>
            <p id="engine-change-description">
              This will end the current session. It will stay saved, and your
              next message will begin a new session with the selected engine.
            </p>
            <div className="engine-change-dialog__actions">
              <button
                className="button button--secondary"
                type="button"
                onClick={() => setPendingEngineAction(null)}
                autoFocus
              >
                Cancel
              </button>
              <button
                className="button button--primary"
                type="button"
                onClick={confirmEngineChange}
              >
                Change engine &amp; start new session
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  );

}
