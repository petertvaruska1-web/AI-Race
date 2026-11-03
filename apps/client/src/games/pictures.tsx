import { STORY_HEROES, STORY_SITUATIONS, STORY_WORDS } from "@airace/protocol";

// How the games draw their content: pattern terms as discs, animals and tiles; Story Dice
// faces as pictures. The words the AI chose are always shown too (R1: nothing is hidden).

export const COLOR_HEX: Record<string, string> = {
  red: "#ef4444",
  blue: "#3b82f6",
  green: "#22c55e",
  yellow: "#eab308",
  pink: "#ec4899",
  purple: "#a855f7",
  orange: "#f97316",
  black: "#1f2937",
  white: "#f9fafb",
  brown: "#92400e",
  gray: "#9ca3af",
  gold: "#d4a017",
};

export const ANIMAL_EMOJI: Record<string, string> = {
  cat: "🐱",
  dog: "🐶",
  cow: "🐮",
  pig: "🐷",
  hen: "🐔",
  fox: "🦊",
  frog: "🐸",
  duck: "🦆",
  bear: "🐻",
  lion: "🦁",
  fish: "🐟",
  bird: "🐦",
};

const HERO_EMOJI = [
  "🦊",
  "🤖",
  "🧓",
  "🐉",
  "🐶",
  "🐭",
  "🐻",
  "🎨",
  "🫖",
  "👧",
  "🐐",
  "⛵",
  "🐌",
  "🐝",
  "👻",
  "🐦",
  "👦",
  "🌙",
  "🐳",
  "🧙",
];
const SITUATION_EMOJI = [
  "🧭",
  "🚪",
  "🌊",
  "🌧️",
  "🤝",
  "🎶",
  "🏁",
  "🗺️",
  "🚆",
  "🏠",
  "🎵",
  "❄️",
  "⭐",
  "🌱",
  "🎩",
  "🏙️",
  "🤫",
  "🎂",
  "🌈",
  "😴",
];
const WORD_EMOJI = ["🔑", "🗺️", "🎩", "🍰", "⭐", "⛵", "🎵", "🚪", "🌳", "🌙", "🔔", "💍"];

export const DICE = {
  hero: STORY_HEROES.map((label, i) => ({ label, emoji: HERO_EMOJI[i] ?? "🎲" })),
  situation: STORY_SITUATIONS.map((label, i) => ({ label, emoji: SITUATION_EMOJI[i] ?? "🎲" })),
  word: STORY_WORDS.map((label, i) => ({ label, emoji: WORD_EMOJI[i] ?? "🎲" })),
};

export type Pic = { kind: "color" | "animal" | "number" | "letter"; value: string };

/** One pattern term as a picture: a colored disc, an animal, or a number or letter tile. */
export function Picture({ p, size = 44 }: { p: Pic; size?: number }) {
  if (p.kind === "color") {
    return (
      <span
        role="img"
        aria-label={p.value}
        title={p.value}
        className="inline-block rounded-full border-2 border-white/20 shadow"
        style={{ width: size, height: size, backgroundColor: COLOR_HEX[p.value] ?? "#888" }}
      />
    );
  }
  if (p.kind === "animal") {
    return (
      <span role="img" aria-label={p.value} title={p.value} style={{ fontSize: size * 0.8 }}>
        {ANIMAL_EMOJI[p.value] ?? "❓"}
      </span>
    );
  }
  return (
    <span
      className="inline-flex items-center justify-center rounded-lg bg-surface-2 font-display font-bold"
      style={{ minWidth: size, height: size, fontSize: size * 0.42, padding: "0 6px" }}
    >
      {p.value}
    </span>
  );
}
