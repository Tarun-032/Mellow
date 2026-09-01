/** Shared form controls for Settings and onboarding. */

import { type ReactNode } from "react";

export const API = "http://127.0.0.1:8765";

export type Preset = {
  label: string;
  base_url: string;
  requires_key: boolean;
  local: boolean;
  /** Only ElevenLabs has one - see TTS_PRESETS. */
  model?: string;
};

/** Wire format shared by llm / stt / tts. */
export type Mode = "local" | "cloud" | "agent";
export type AgentSpeed = "fast" | "balanced" | "deep";

export type Transport = {
  mode: Mode;
  provider: string;
  base_url: string;
  api_key: string;
  has_api_key: boolean;
  model: string;
};

export type Capability = "llm" | "stt" | "tts";

/** One detected coding-agent CLI. */
export type AgentInfo = {
  id: string;
  label: string;
  installed: boolean;
  path: string;
  install: string;
  vision: boolean;
  signed_in: boolean;
  auth_detail: string;
  /** Models the CLI flag is known to accept. */
  models: Record<string, string>;
};

/** Which button is running. Only one action at a time makes sense. */
export type Action = "save" | Capability | "voices" | "connect";

export type NoticeData = { where: Action; kind: "ok" | "error"; text: string };

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || `request failed (${response.status})`);
  }
  return body as T;
}

/** Label above, helper below. */
export function Field(props: {
  label: ReactNode;
  optional?: boolean;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <>
      <label>
        {/* Caption row with its marker. */}
        <span className="field-label">
          {props.label}
          {props.optional && <span className="optional">optional</span>}
        </span>
        {props.children}
      </label>
      {props.hint !== undefined && <p className="field-note">{props.hint}</p>}
    </>
  );
}

/** A test's outcome, attached to the section that produced it. */
export function Notice(props: { kind: "ok" | "error"; children: ReactNode }) {
  return (
    <p className={`notice notice--${props.kind}`} role="status" aria-live="polite">
      {props.children}
    </p>
  );
}

/** Local vs cloud radios (arrow-key friendly). */
export function ModeToggle(props: {
  name: string;
  value: string;
  entries: readonly [string, string, string][];
  onChange: (value: string) => void;
}) {
  return (
    <fieldset className="modes">
      <legend className="modes__legend">Where this runs</legend>
      {props.entries.map(([mode, label, hint]) => (
        <label className="mode" key={mode}>
          <input
            type="radio"
            name={`${props.name}-mode`}
            value={mode}
            checked={props.value === mode}
            onChange={() => props.onChange(mode)}
          />
          <span className="mode__label">{label}</span>
          <span className="mode__hint">{hint}</span>
        </label>
      ))}
    </fieldset>
  );
}

/** Provider, base URL, model, and key fields. */
export function CloudFields(props: {
  name: Capability;
  section: Transport;
  presets: Record<string, Preset>;
  modelHint: string;
  onPatch: (fields: Partial<Transport>) => void;
  onProvider: (provider: string) => void;
}) {
  const { section, presets } = props;
  const preset = presets[section.provider];
  return (
    <>
      <label>
        <span className="field-label">Provider</span>
        <select
          value={section.provider}
          onChange={(event) => props.onProvider(event.target.value)}
        >
          {/* Local providers belong to the On device half. */}
          {Object.entries(presets)
            .filter(([, item]) => !item.local)
            .map(([key, item]) => (
              <option key={key} value={key}>
                {item.label}
              </option>
            ))}
        </select>
      </label>

      <label>
        <span className="field-label">Base URL</span>
        <input
          type="url"
          value={section.base_url}
          onChange={(event) => props.onPatch({ base_url: event.target.value })}
          readOnly={section.provider !== "custom"}
          spellCheck={false}
        />
      </label>

      <label>
        <span className="field-label">Model name</span>
        <input
          required
          value={section.model}
          onChange={(event) => props.onPatch({ model: event.target.value })}
          placeholder={props.modelHint}
          spellCheck={false}
        />
      </label>

      <label>
        <span className="field-label">
          API key {preset?.requires_key ? <span aria-hidden="true">*</span> : null}
        </span>
        <input
          type="password"
          value={section.api_key}
          onChange={(event) => props.onPatch({ api_key: event.target.value })}
          placeholder={
            section.has_api_key
              ? "Saved. Leave blank to keep it"
              : "Not required for local servers"
          }
          autoComplete="off"
        />
      </label>
      {section.provider === "custom" && (
        <p className="field-note">
          The key and everything Mellow sends will go to this URL.
        </p>
      )}
    </>
  );
}

/** Agent CLI + model + connect/test. */
export function AgentFields(props: {
  provider: string;
  model: string;
  speed: AgentSpeed;
  agents: AgentInfo[];
  busy: string | null;
  onPatch: (fields: {
    provider?: string;
    model?: string;
    agent_speed?: AgentSpeed;
  }) => void;
  onConnect: () => void;
  notice: ReactNode;
}) {
  const { agents, busy } = props;
  const selected = agents.find((item) => item.id === props.provider);
  const known = Object.entries(selected?.models ?? {});
  const exactModelAvailable =
    !props.model || known.some(([value]) => value === props.model);
  const unavailableValue = "__mellow_unavailable_model__";
  return (
    <>
      <label>
        Agent
        <select
          value={props.provider}
          onChange={(event) =>
            props.onPatch({ provider: event.target.value, model: "" })
          }
        >
          {/* Keep an unknown saved provider visible. */}
          {!agents.some((item) => item.id === props.provider) && (
            <option value={props.provider}>{props.provider} - not found</option>
          )}
          {agents.map((item) => (
            <option key={item.id} value={item.id}>
              {item.label}
              {item.installed ? "" : " · not installed"}
            </option>
          ))}
        </select>
      </label>

      <label>
        Model
        <select
          value={exactModelAvailable ? props.model : unavailableValue}
          onChange={(event) => props.onPatch({ model: event.target.value })}
        >
          <option value="">Agent default · may change with your plan</option>
          {/* Don't echo a retired slug; stay neutral. */}
          {!exactModelAvailable && (
            <option value={unavailableValue} disabled>
              Choose an available model
            </option>
          )}
          {known.map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </label>
      {!exactModelAvailable && (
        <p className="field-note">
          The previously selected model is no longer available for exact use.
          Choose another model before saving; Mellow will not substitute one.
        </p>
      )}

      <label>
        Response speed
        <select
          value={props.speed}
          onChange={(event) =>
            props.onPatch({ agent_speed: event.target.value as AgentSpeed })
          }
        >
          <option value="fast">Fast · lowest latency</option>
          <option value="balanced">Balanced · more reasoning</option>
          <option value="deep">Deep · most reasoning</option>
        </select>
      </label>

      <div className="history-bar">
        <button
          className="button button--secondary"
          type="button"
          disabled={busy !== null}
          onClick={props.onConnect}
        >
          {busy === "connect" ? "Checking…" : "Connect"}
        </button>
      </div>
      {props.notice}

      {selected && !selected.installed && (
        <p className="field-note">
          Not installed yet. In any terminal, run: <code>{selected.install}</code>
          , then pick it again here.
        </p>
      )}
      {selected?.installed && (
        <p className="field-note">
          {selected.signed_in ? "Signed in" : "Not signed in"}
          {selected.auth_detail ? ` · ${selected.auth_detail}` : ""}
        </p>
      )}
      {/* One helper line (was two paragraphs). */}
      <p className="field-note">
        No API key. It signs in with your subscription, gets no tools, and
        takes around five seconds an answer.
      </p>
    </>
  );
}
