import { BLOCKED_WORDS } from "./wordlist.ts";

// What a notebook may hold (game-layer design §10). It is private training text, so the
// filter only blocks: it never changes a word of what the player wrote.

export const MAX_NOTEBOOK_WORDS = 5_000;

const LINK = /\b(?:https?:\/\/|www\.)\S+/i;
const EMAIL = /[\w.+-]+@[\w-]+\.[a-z]{2,}/i;
/** A run of digits with phone-like separators; it counts only with 9 or more digits. */
const PHONE_LIKE = /\+?\(?\d[\d ().-]{6,}\d/g;
const BLOCKED = new RegExp(`\\b(?:${BLOCKED_WORDS.join("|")})(?:s|es|ed|er|ers|ing|in)?\\b`, "i");

/** How many words `text` has (runs of non-space). */
export function wordCount(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

function hasPhone(text: string): boolean {
  for (const m of text.matchAll(PHONE_LIKE)) {
    if ((m[0].match(/\d/g)?.length ?? 0) >= 9) return true;
  }
  return false;
}

/**
 * Why players' text can't be kept, or null when it can: links, email addresses, phone numbers,
 * the AI's own characters and blocked words. Shared by notebooks and the world chat; it only
 * blocks, never changes a word. `kind` names the text in the refusal.
 */
export function textProblem(
  text: string,
  kind: { many: string; one: string } = { many: "Messages", one: "That message" },
): string | null {
  if (LINK.test(text) || EMAIL.test(text) || hasPhone(text)) {
    return `${kind.many} can't hold links, email addresses or phone numbers.`;
  }
  if (text.includes("<|")) return "That text has characters the AI uses for itself.";
  if (BLOCKED.test(text)) return `${kind.one} has words that aren't allowed.`;
  return null;
}

/** Why a notebook can't hold `text`, or null when it can (an empty one clears it). */
export function notebookProblem(text: string): string | null {
  if (wordCount(text) > MAX_NOTEBOOK_WORDS) return "Notebooks hold up to 5,000 words.";
  return textProblem(text, { many: "Notebooks", one: "That notebook" });
}

/** The notebook as training documents: its paragraphs (blank lines apart), trimmed. */
export function notebookParagraphs(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);
}
