import type { DatasetId, Speed, TrainingPlan } from "@airace/protocol";

export type BalanceVersion = "balance-v1" | "balance-test";
export type BrainSize = { id: string; nLayer: number; dModel: number };
export type LengthOption = { id: string; tokens: number };

/** Every tunable number of the economy. A world records its version (replay). */
export type Balance = {
  version: BalanceVersion;
  /** Real milliseconds per game day at speed 1. */
  tickMsAt1x: number;
  speeds: readonly Speed[];
  foundingCash: number;
  /** Rented cloud compute per game day, in PFLOP (1e15 operations). */
  cloudPflopsPerDay: number;
  /** Dollars per PFLOP of rented compute. */
  cloudPricePerPflop: number;
  factCheckCost: number;
  maxMembers: number;
  brainSizes: readonly BrainSize[];
  attentionSpans: readonly number[];
  lengths: readonly LengthOption[];
  defaultPlan: TrainingPlan;
};

const EVEN_MIX: Record<DatasetId, number> = {
  web: 12,
  books: 12,
  educational: 12,
  conversations: 12,
  code: 12,
  reasoning: 12,
  facts: 12,
  creative: 12,
};

function defaultPlan(brainSize: string, attentionSpan: number, length: string): TrainingPlan {
  return {
    mix: { ...EVEN_MIX },
    cleaning: "standard",
    removeDuplicates: false,
    factCheck: false,
    variety: "natural",
    brainSize,
    attentionSpan,
    length,
    learningStyle: 0.5,
    review: 0,
    probePrompt: "",
    parentVersionId: null,
  };
}

// The first model (tiny, standard length) is 0.104 PFLOP: 13 game days
// (104 s at 1x) and $41,596, about a sixth of the founding cash.
export const BALANCE_V1: Balance = {
  version: "balance-v1",
  tickMsAt1x: 8000,
  speeds: [1, 2, 4],
  foundingCash: 250_000,
  cloudPflopsPerDay: 0.008,
  cloudPricePerPflop: 400_000,
  factCheckCost: 5000,
  maxMembers: 8,
  brainSizes: [
    { id: "tiny", nLayer: 4, dModel: 128 },
    { id: "small", nLayer: 6, dModel: 192 },
    { id: "medium", nLayer: 8, dModel: 256 },
  ],
  attentionSpans: [128, 256, 512],
  lengths: [
    { id: "short", tokens: 6_291_456 },
    { id: "standard", tokens: 12_582_912 },
    { id: "long", tokens: 25_165_824 },
  ],
  defaultPlan: defaultPlan("tiny", 256, "standard"),
};

/** Fast worlds for end-to-end and quick tests. */
export const BALANCE_TEST: Balance = {
  ...BALANCE_V1,
  version: "balance-test",
  tickMsAt1x: 200,
  brainSizes: [{ id: "tiny", nLayer: 2, dModel: 64 }],
  attentionSpans: [64],
  lengths: [{ id: "short", tokens: 65_536 }],
  cloudPflopsPerDay: 0.000005, // the 2x64 test model trains in 30 game days (6 s at 1x)
  defaultPlan: defaultPlan("tiny", 64, "short"),
};

export function getBalance(version: BalanceVersion): Balance {
  return version === "balance-v1" ? BALANCE_V1 : BALANCE_TEST;
}
