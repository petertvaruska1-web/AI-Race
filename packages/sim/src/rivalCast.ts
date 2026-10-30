import {
  type Archetype,
  LOGO_SHAPES,
  LOGO_SYMBOLS,
  type Logo,
  type Palette,
  type RivalStyle,
} from "@airace/protocol";
import { PALETTE_PRESETS } from "./palette.ts";
import { type RngState, rngInt } from "./rng.ts";

// The rivals' names, looks and products (game-layer design §9): invented words,
// none a real AI company, drawn per world from the world rng.

export const RIVAL_NAME_FIRST = [
  "Nova",
  "Quanta",
  "Lumen",
  "Vertex",
  "Cinder",
  "Mosaic",
  "Orbit",
  "Pike",
  "Harbor",
  "Flux",
  "Ember",
  "Zephyr",
  "Atlas",
  "Kestrel",
  "Juniper",
  "Onyx",
  "Prism",
  "Tidal",
  "Sable",
  "Corvid",
  "Halo",
  "Ridge",
  "Aurora",
  "Basalt",
] as const;

export const RIVAL_NAME_ENDINGS: Record<Archetype | RivalStyle, readonly string[]> = {
  giant: ["Global", "Dynamics", "Systems"],
  upstart: ["Labs", "AI", "Works"],
  lab: ["Institute", "Research", "Foundry"],
  premium: ["Intelligence", "Systems", "Analytics"],
  budget: ["AI", "Go", "Mind"],
  specialist: ["Labs", "Logic", "Studio"],
  consumer: ["AI", "Chat", "Pal"],
};

export const PRODUCT_WORDS: Record<RivalStyle, readonly string[]> = {
  premium: ["Titan", "Apex", "Sage"],
  budget: ["Lite", "Go", "Spark"],
  specialist: ["Forge", "Scholar", "Muse"],
  consumer: ["Buddy", "Chat", "Pal"],
};

/** A rival's k-th product (from 1): its word, numbered from the second release on. */
export function rivalProductName(style: RivalStyle, word: number, k: number): string {
  const words = PRODUCT_WORDS[style];
  const base = words[word % words.length] ?? "Model";
  return k === 1 ? base : `${base} ${k}`;
}

export type CastLook = { name: string; logo: Logo; palette: Palette; productWord: number };

/**
 * A rival's name, logo and palette. `taken` names (lowercase) and `takenPrimaries`
 * are skipped; both sets are updated with the result.
 */
export function drawLook(
  rng: RngState,
  ending: Archetype | RivalStyle,
  taken: Set<string>,
  takenPrimaries: Set<string>,
): { look: CastLook; rng: RngState } {
  let r = rng;
  const draw = (n: number) => {
    const x = rngInt(r, n);
    r = x.next;
    return x.value;
  };
  const endings = RIVAL_NAME_ENDINGS[ending];
  const candidates = RIVAL_NAME_FIRST.flatMap((first) =>
    endings.map((e) => `${first} ${e}`),
  ).filter((n) => !taken.has(n.toLowerCase()));
  const name = candidates[draw(candidates.length)] ?? `Rival ${taken.size + 1}`;
  taken.add(name.toLowerCase());
  const palettes = PALETTE_PRESETS.filter(
    (p) => !takenPrimaries.has(p.palette.primary.toLowerCase()),
  );
  const palette = (palettes[draw(palettes.length)] ?? PALETTE_PRESETS[0])?.palette as Palette;
  takenPrimaries.add(palette.primary.toLowerCase());
  const shape = LOGO_SHAPES[draw(LOGO_SHAPES.length)] ?? "circle";
  const symbol = LOGO_SYMBOLS[draw(LOGO_SYMBOLS.length)] ?? "spark";
  const productWord = draw(3);
  return {
    look: { name, logo: { shape, symbol, initials: "" }, palette: { ...palette }, productWord },
    rng: r,
  };
}

export type TrashEvent =
  | "playerLaunch"
  | "playerWinsBattle"
  | "playerPassesArch"
  | "playerRescued"
  | "archLaunch";

/** The arch-rival founder's lines, in its archetype's voice (game text, never an AI's words). */
export const TRASH_TALK: Record<Archetype, Record<TrashEvent, readonly [string, string]>> = {
  giant: {
    playerLaunch: [
      "Cute launch, {player}. Call us when {product} can count.",
      "{product}? We had something like that in a drawer years ago.",
    ],
    playerWinsBattle: [
      "One battle. We have a whole building of engineers who will remember it, {player}.",
      "Beginner's luck. Enjoy it while {player} still can.",
    ],
    playerPassesArch: [
      "More customers than us? Temporary. Everything about {player} is.",
      "Let {player} have the crowd. We have the contracts.",
    ],
    playerRescued: [
      "Out of money already, {player}? Our coffee budget is bigger.",
      "Rescued. That is the word we use for small companies, {player}.",
    ],
    archLaunch: [
      "{product} is here. The rest of you can stop trying now.",
      "We launched {product}. Our competitors launched excuses.",
    ],
  },
  upstart: {
    playerLaunch: [
      "Welcome to the party, {player}. We brought snacks; you brought {product}.",
      "{product}, huh? Bold name. Let's see if it's bold at anything else.",
    ],
    playerWinsBattle: [
      "Okay, okay, {player} won that one. Rematch. Loser buys pizza.",
      "Fine, {player}. Even our AI needs a nap sometimes.",
    ],
    playerPassesArch: [
      "{player} has more customers than us? We're writing that on the fridge as motivation.",
      "Congrats, {player}. Now everybody's watching you, not us. Perfect.",
    ],
    playerRescued: [
      "Hey {player}, need a loan? Just kidding, so do we.",
      "Rough month, {player}? Happens to the best of us. And to you.",
    ],
    archLaunch: [
      "{product} just dropped and it's honestly kind of amazing. We'd know.",
      "New AI who dis? {product} is live.",
    ],
  },
  lab: {
    playerLaunch: [
      "{product}. Interesting. We have notes.",
      "We observed the launch of {product}. The data will speak.",
    ],
    playerWinsBattle: [
      "A result is a result. We are already studying it, {player}.",
      "{player} won this round. We are measuring why.",
    ],
    playerPassesArch: [
      "Customers are a noisy measure. {player} will learn this.",
      "We note that {player} is ahead. We note everything.",
    ],
    playerRescued: [
      "Funding is a hypothesis. {player}'s was disproved.",
      "We wish {player} well. Our models predicted this.",
    ],
    archLaunch: [
      "{product} leaves the lab today. You will not understand it at first.",
      "We release {product}. Ask it something difficult.",
    ],
  },
};
