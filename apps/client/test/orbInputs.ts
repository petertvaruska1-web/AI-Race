import type { OrbInput } from "../src/orb/params.ts";

// The orb inputs the orb and 3D core tests share: no AI yet, and a scored first AI.

export const PALETTE = { primary: "#4f8cff", secondary: "#ffb020" };
export const SCORES = {
  language: 60,
  reasoning: 8,
  pattern: 80,
  knowledge: 44,
  coding: 38,
  creativity: 2,
  consistency: 37,
  instruction: 63,
};
export const TRAITS = {
  verbosity: 18,
  inventiveness: 0.5,
  steadiness: 0.4,
  boldness: 0.6,
  slip_rate: 0.2,
  warmth: 0.3,
};

export const NONE: OrbInput = {
  overall: null,
  scores: null,
  traits: null,
  trainedTokens: 0,
  palette: PALETTE,
  training: null,
};
export const SCORED: OrbInput = {
  overall: 42,
  scores: SCORES,
  traits: TRAITS,
  trainedTokens: 12_582_912,
  palette: PALETTE,
  training: null,
};
