/**
 * Meeting files, built here rather than fetched.
 *
 * `GET /meetings/{id}/export` always returns notes AND transcript together and
 * gives byte-identical output for md and txt, so it cannot answer "just the
 * transcript, as plain text". Everything needed is already in the loaded
 * meeting, so the file is assembled from that. The route is untouched.
 *
 * Kept free of JSX so `node scripts/export.check.ts` can import it directly.
 */
// Explicit extension: `node scripts/export.check.ts` resolves this without a bundler.
import { inlineSpans, parseNotes } from "./notes.ts";

export type Content = "transcript" | "notes";
export type Format = "md" | "txt" | "json";

export type Meeting = { id: string; title: string; created: string; engine: string; notes: string };
export type Turn = { speaker: string; text: string };

const MIME: Record<Format, string> = {
  md: "text/markdown;charset=utf-8",
  txt: "text/plain;charset=utf-8",
  json: "application/json",
};

/** Filenames travel between machines, so keep them to a safe ASCII slug. */
function slug(title: string) {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60);
}

/** Markdown down to readable plain text, reusing the renderer's own parser. */
function plain(markdown: string) {
  return parseNotes(markdown).map(block => {
    const flat = (text: string) => inlineSpans(text).map(span => span.text).join("");
    if (block.kind === "list") {
      return block.items.map(item => `${"  ".repeat(item.depth)}- ${flat(item.text)}`).join("\n");
    }
    return flat(block.text);
  }).join("\n\n");
}

function transcript(turns: Turn[], heading: boolean) {
  const lines = turns.flatMap(turn => [`${turn.speaker}:`, turn.text, ""]);
  return heading ? ["## Transcript", "", ...lines] : lines;
}

export function buildExport(meeting: Meeting, turns: Turn[], content: Content, format: Format) {
  const name = slug(meeting.title) || meeting.id;
  const file = { filename: `mellow-${name}-${content}.${format}`, mime: MIME[format] };

  if (format === "json") {
    const body = content === "notes"
      ? { title: meeting.title, created: meeting.created, engine: meeting.engine, notes: meeting.notes }
      : { title: meeting.title, created: meeting.created, segments: turns };
    return { ...file, text: JSON.stringify(body, null, 2) };
  }

  const markdown = format === "md";
  const head = markdown ? [`# ${meeting.title}`, "", meeting.created, ""] : [meeting.title, meeting.created, ""];
  const body = content === "notes"
    ? [markdown ? meeting.notes : plain(meeting.notes), ""]
    : transcript(turns, markdown);
  return { ...file, text: [...head, ...body].join("\n") };
}
