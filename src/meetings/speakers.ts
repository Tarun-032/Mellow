export type Speakers = Record<string, { key: string; name: string; merged_into: string | null }>;

export function canonical(key: string, speakers: Speakers = {}): string {
  const original = key, seen = new Set<string>();
  while (speakers[key]?.merged_into) {
    if (seen.has(key)) return original;
    seen.add(key);
    key = speakers[key].merged_into!;
    if (!speakers[key]) return original;
  }
  return key;
}

export function speakerLabel(key: string, speakers: Speakers = {}): string {
  key = canonical(key, speakers);
  const name = speakers[key]?.name || key;
  if (name === "Other participants") return "Other participant";
  const duplicate = Object.entries(speakers).filter(([k, v]) => !v.merged_into && (v.name || k) === name).length > 1;
  return duplicate && name !== key ? `${name} (${key})` : name;
}
