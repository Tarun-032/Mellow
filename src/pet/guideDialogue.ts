export const GUIDE_DIALOGUE_KEY = "mellow.guide-dialogue";

export type GuideDialogue = {
  text: string;
  error: boolean;
  side: "left" | "right";
  lift: "above" | "below";
};

export function readGuideDialogue(): GuideDialogue | null {
  try {
    const value = JSON.parse(localStorage.getItem(GUIDE_DIALOGUE_KEY) ?? "null");
    return value && typeof value.text === "string" ? value : null;
  } catch {
    return null;
  }
}
