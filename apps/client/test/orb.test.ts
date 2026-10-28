import { describe, expect, it } from "vitest";
import { type OrbInput, orbParams } from "../src/orb/params.ts";

const PALETTE = { primary: "#4f8cff", secondary: "#ffb020" };
const SCORES = {
  language: 60,
  reasoning: 8,
  pattern: 80,
  knowledge: 44,
  coding: 38,
  creativity: 2,
  consistency: 37,
  instruction: 63,
};
const TRAITS = {
  verbosity: 18,
  inventiveness: 0.5,
  steadiness: 0.4,
  boldness: 0.6,
  slip_rate: 0.2,
  warmth: 0.3,
};

const NONE: OrbInput = {
  overall: null,
  scores: null,
  traits: null,
  trainedTokens: 0,
  palette: PALETTE,
  training: null,
};
const SCORED: OrbInput = {
  overall: 42,
  scores: SCORES,
  traits: TRAITS,
  trainedTokens: 12_582_912,
  palette: PALETTE,
  training: null,
};

describe("orb parameters", () => {
  it("is a faint spark before the first AI", () => {
    const p = orbParams(NONE);
    expect(p.radius).toBeLessThan(0.3);
    expect(p.rings).toBe(0);
    expect(p.intake).toBeNull();
  });

  it("grows bigger and brighter with the overall score", () => {
    let last = orbParams({ ...SCORED, overall: 0 });
    for (const overall of [10, 25, 42, 60, 80, 100]) {
      const p = orbParams({ ...SCORED, overall });
      expect(p.radius).toBeGreaterThan(last.radius);
      expect(p.brightness).toBeGreaterThan(last.brightness);
      last = p;
    }
    expect(last.radius).toBeLessThanOrEqual(1);
  });

  it("has a ring for each skill at 20 or more, plus rings for the words it has read", () => {
    const scores = { language: 60, reasoning: 5, pattern: 80, coding: 2 };
    const read = (tokens: number) => orbParams({ ...SCORED, scores, trainedTokens: tokens }).rings;
    expect(read(1_000_000)).toBeCloseTo(1 + 2, 9);
    expect(read(10_000_000)).toBeCloseTo(1 + 2 + 1, 9);
    expect(read(100_000_000)).toBeCloseTo(1 + 2 + 2, 9);
  });

  it("swirls faster when bolder and moves more smoothly when steadier", () => {
    const at = (traits: Partial<typeof TRAITS>) =>
      orbParams({ ...SCORED, traits: { ...TRAITS, ...traits } });
    expect(at({ boldness: 0.9 }).swirl).toBeGreaterThan(at({ boldness: 0.1 }).swirl);
    expect(at({ steadiness: 0.9 }).smoothness).toBeGreaterThan(at({ steadiness: 0.1 }).smoothness);
    expect(at({ verbosity: 60 }).sparkRate).toBeGreaterThan(at({ verbosity: 5 }).sparkRate);
    expect(at({ slip_rate: 0.6 }).flicker).toBeGreaterThan(at({ slip_rate: 0 }).flicker);
  });

  it("uses the company's colors, primary first", () => {
    const p = orbParams(SCORED);
    expect(p.colors[0]).toBe(PALETTE.primary);
    expect(p.colors.at(-1)).toBe(PALETTE.secondary);
  });

  it("draws data in only while training, brighter as the loss falls", () => {
    const training = {
      progress: 0.4,
      lossDrop: 0.2,
      mix: {
        web: 50,
        books: 50,
        educational: 0,
        conversations: 0,
        code: 0,
        reasoning: 0,
        facts: 0,
        creative: 0,
      },
    };
    expect(orbParams(SCORED).intake).toBeNull();
    const p = orbParams({ ...NONE, training });
    expect(p.intake?.colors).toHaveLength(2);
    expect(p.intake?.shares).toEqual([0.5, 0.5]);
    expect(
      orbParams({ ...NONE, training: { ...training, lossDrop: 0.8 } }).brightness,
    ).toBeGreaterThan(p.brightness);
  });

  it("changes no parameter by more than 10% for a one-point score change", () => {
    const numbers = (input: OrbInput) =>
      Object.entries(orbParams(input)).filter(
        (e): e is [string, number] => typeof e[1] === "number",
      );
    for (const category of Object.keys(SCORES) as (keyof typeof SCORES)[]) {
      for (const shift of [-1, 1]) {
        const moved = {
          ...SCORED,
          overall: (SCORED.overall ?? 0) + shift,
          scores: { ...SCORES, [category]: SCORES[category] + shift },
        };
        const after = new Map(numbers(moved));
        for (const [name, before] of numbers(SCORED)) {
          const now = after.get(name) ?? Number.NaN;
          expect(Math.abs(now - before), `${name} after ${category} ${shift}`).toBeLessThanOrEqual(
            0.1 * Math.max(Math.abs(before), 0.05),
          );
        }
      }
    }
    // Right at the edge where a skill earns its ring, too.
    const edge = (x: number) => orbParams({ ...SCORED, scores: { ...SCORES, reasoning: x } }).rings;
    for (let x = 14; x <= 26; x++) {
      expect(Math.abs(edge(x + 1) - edge(x))).toBeLessThanOrEqual(0.1 * edge(x));
    }
  });
});
