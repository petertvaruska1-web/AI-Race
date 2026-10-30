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
