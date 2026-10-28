import { DATASET_IDS, type DatasetId, type Palette } from "@airace/protocol";
import { mix } from "../theme/palette.ts";

// The AI as a glowing orb (game-layer design §4). Everything about how it looks
// comes from real measurements of the version, through smooth functions: no
// tiers, no unlocks (R5). A better AI is a bigger, brighter, more intricate orb.

export type OrbInput = {
  /** The overall bench score, 0..100; null before scoring. */
  overall: number | null;
  scores: Record<string, number | null> | null;
  /** Measured style traits (the fingerprint); null before scoring. */
  traits: Record<string, number> | null;
  /** Words (tokens) read over the version's whole lineage; 0 when there is no AI yet. */
  trainedTokens: number;
  palette: Palette;
  /** While a run trains: how far it is, how far the loss has fallen (0..1), and its data mix. */
  training: { progress: number; lossDrop: number; mix: Record<DatasetId, number> } | null;
};

export type OrbParams = {
  /** Core radius, as a share of the largest (0..1). */
  radius: number;
  brightness: number;
  /** Whole rings, plus a fraction: the next ring fading in. */
  rings: number;
  ringDetail: number;
  swirl: number;
  turbulence: number;
  smoothness: number;
  sparkRate: number;
  glowSoftness: number;
  flicker: number;
  /** The company's primary color through to its secondary. */
  colors: string[];
  /** Streams of data flowing in while training: a color and a share per dataset in the mix. */
  intake: { rate: number; colors: string[]; shares: number[] } | null;
};

/** One color per dataset, for the streams flowing in during training. */
export const DATASET_COLORS: Record<DatasetId, string> = {
  web: "#60a5fa",
  books: "#f59e0b",
  educational: "#34d399",
  conversations: "#f472b6",
  code: "#a78bfa",
  reasoning: "#22d3ee",
  facts: "#facc15",
  creative: "#fb7185",
};

/** Traits of a typical AI, used until the first scoring. */
const NEUTRAL: Record<string, number> = {
  verbosity: 15,
  inventiveness: 0.5,
  steadiness: 0.5,
  boldness: 0.5,
  slip_rate: 0.1,
  warmth: 0.2,
};

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

/** 0 below 15, 1 above 25, smooth in between: a skill's ring fades in as it nears 20. */
function skillRing(score: number): number {
  const t = clamp01((score - 15) / 10);
  return t * t * (3 - 2 * t);
}

export function orbParams(input: OrbInput): OrbParams {
  const version = input.trainedTokens > 0;
  const q = clamp01((input.overall ?? 0) / 100);
  const trait = (name: string) => {
    const v = input.traits?.[name];
    return typeof v === "number" && Number.isFinite(v) ? v : (NEUTRAL[name] ?? 0);
  };
  const bold = clamp01(trait("boldness"));
  const inventive = clamp01(trait("inventiveness"));
  const words = Math.max(0, trait("verbosity"));
  const warmth = Math.max(0, trait("warmth"));
  const millions = input.trainedTokens / 1e6;

  let skills = 0;
  for (const s of Object.values(input.scores ?? {}))
    if (typeof s === "number") skills += skillRing(s);

  const lossDrop = clamp01(input.training?.lossDrop ?? 0);
  const brightness = (version ? 0.35 + 0.65 * q : 0.25) + (input.training ? 0.35 * lossDrop : 0);

  const p = input.palette;
  const colors = [0, 0.25, 0.5, 0.75, 1].map((k) =>
    k === 0 ? p.primary : k === 1 ? p.secondary : mix(p.secondary, p.primary, k),
  );

  let intake: OrbParams["intake"] = null;
  if (input.training) {
    const used = DATASET_IDS.filter((id) => (input.training?.mix[id] ?? 0) > 0);
    const total = used.reduce((sum, id) => sum + (input.training?.mix[id] ?? 0), 0);
    intake = {
      rate: 0.35 + 0.65 * clamp01(input.training.progress),
      colors: used.map((id) => DATASET_COLORS[id]),
      shares: used.map((id) => (input.training?.mix[id] ?? 0) / total),
    };
  }

  return {
    radius: version ? 0.25 + 0.6 * q : 0.14,
    brightness: Math.min(1, brightness),
    rings: version ? 1 + skills + Math.log10(Math.max(1, millions)) : 0,
    ringDetail: clamp01(0.15 + 0.3 * Math.log10(1 + millions)),
    swirl: 0.15 + 0.85 * (0.65 * bold + 0.35 * inventive),
    turbulence: 0.1 + 0.9 * (0.7 * inventive + 0.3 * bold),
    smoothness: 0.15 + 0.85 * clamp01(trait("steadiness")),
    sparkRate: words / (words + 25),
    glowSoftness: 0.2 + (0.8 * warmth) / (warmth + 0.4),
    flicker: 0.7 * clamp01(trait("slip_rate")),
    colors,
    intake,
  };
}
