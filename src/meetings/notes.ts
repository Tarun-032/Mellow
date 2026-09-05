/**
 * Meeting notes arrive as free Markdown — `mellowd/meeting_notes.py` asks the model
 * for "concise Markdown" and nothing post-processes the answer. Parse it here into
 * plain data so the renderer can build React nodes: no HTML string ever exists, so
 * model-authored text has nowhere to inject markup.
 *
 * Kept free of JSX so `node scripts/notes.check.ts` can import it directly.
 */

export type Block =
  | { kind: "heading"; text: string }
  | { kind: "para"; text: string }
  | { kind: "list"; ordered: boolean; items: { depth: number; text: string }[] };

export type Span = { text: string; bold?: boolean; time?: boolean };

const HEADING = /^#{1,6}\s+(.*)$/;
const BULLET = /^(\s*)([*+-]|\d+[.)])\s+(.*)$/;
/** A line that is nothing but bold text. Models write section titles this way. */
const BOLD_LINE = /^\*\*([^*]+)\*\*:?$/;
/** `**bold**` runs, and bracketed stamps like [00:04:12] or [00:00:57 - 00:01:34]. */
const INLINE = /\*\*([^*]+)\*\*|(\[\d{1,2}:\d{2}(?::\d{2})?(?:\s*[-–—]\s*\d{1,2}:\d{2}(?::\d{2})?)?\])/g;

export function parseNotes(markdown: string): Block[] {
  const blocks: Block[] = [];
  let list: Extract<Block, { kind: "list" }> | null = null;

  for (const raw of markdown.split("\n")) {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim()) { list = null; continue; }

    const bullet = BULLET.exec(line);
    if (bullet) {
      const ordered = !/^[*+-]$/.test(bullet[2]);
      // Tabs and the 2-, 3- and 4-space conventions all reach depth 1 at one level in.
      const depth = Math.min(3, Math.floor(bullet[1].replace(/\t/g, "  ").length / 2));
      if (!list || list.ordered !== ordered) {
        list = { kind: "list", ordered, items: [] };
        blocks.push(list);
      }
      list.items.push({ depth, text: bullet[3].trim() });
      continue;
    }

    list = null;
    const heading = HEADING.exec(line) ?? BOLD_LINE.exec(line);
    if (heading) blocks.push({ kind: "heading", text: heading[1].trim() });
    else blocks.push({ kind: "para", text: line.trim() });
  }
  return blocks;
}

export function inlineSpans(text: string): Span[] {
  const spans: Span[] = [];
  let at = 0;
  for (const match of text.matchAll(INLINE)) {
    const start = match.index;
    if (start > at) spans.push({ text: text.slice(at, start) });
    if (match[1] !== undefined) spans.push({ text: match[1], bold: true });
    else spans.push({ text: match[2], time: true });
    at = start + match[0].length;
  }
  if (at < text.length) spans.push({ text: text.slice(at) });
  return spans.length ? spans : [{ text }];
}
