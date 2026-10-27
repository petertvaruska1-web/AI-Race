import { BENCH_CATEGORIES, GROUP_IDS, REGION_IDS, TrainingPlan } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  adoptionShare,
  BALANCE_TEST,
  BALANCE_V2,
  dateOfDay,
  getBalance,
  paramCount,
  type RngState,
  rngInt,
  rngNext,
  seedRng,
  stableStringify,
  stateHash,
  trainingFlops,
} from "../src/index.ts";

function draw(seed: number, n: number): number[] {
  let s: RngState = seedRng(seed);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const r = rngNext(s);
    out.push(r.value);
    s = r.next;
  }
  return out;
}

describe("PRNG", () => {
  it("repeats for the same seed and stays in [0, 1)", () => {
    const a = draw(42, 1000);
    expect(draw(42, 1000)).toEqual(a);
    for (const x of a) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });

  it("differs between seeds", () => {
    const firsts = new Set([0, 1, 2, 3, 42, 2 ** 31, -1].map((s) => draw(s, 1)[0]));
    expect(firsts.size).toBe(7);
  });

  it("does not change the state it is given and keeps it JSON-safe", () => {
    const s = seedRng(7);
    const copy = structuredClone(s);
    rngNext(s);
    expect(s).toEqual(copy);
    expect(JSON.parse(JSON.stringify(s))).toEqual(s);
  });

  it("spreads values roughly evenly", () => {
    const buckets = new Array(10).fill(0);
    for (const x of draw(9, 10000)) buckets[Math.floor(x * 10)]++;
    for (const b of buckets) expect(b).toBeGreaterThan(850);
  });

  it("draws integers in [0, n)", () => {
    let s = seedRng(5);
    const seen = new Set<number>();
    for (let i = 0; i < 500; i++) {
      const r = rngInt(s, 6);
      expect(Number.isInteger(r.value)).toBe(true);
      seen.add(r.value);
      s = r.next;
    }
    expect([...seen].sort()).toEqual([0, 1, 2, 3, 4, 5]);
    expect(() => rngInt(s, 0)).toThrow();
  });
});

describe("hashing", () => {
  it("ignores key order", () => {
    expect(stateHash({ a: 1, b: [2] })).toBe(stateHash({ b: [2], a: 1 }));
    expect(stableStringify({ b: { y: 1, x: 2 }, a: null })).toBe('{"a":null,"b":{"x":2,"y":1}}');
  });

  it("gives 16 hex characters that change with the value", () => {
    expect(stateHash({ a: 1 })).toMatch(/^[0-9a-f]{16}$/);
    expect(stateHash({ a: 1 })).not.toBe(stateHash({ a: 2 }));
    expect(stateHash([1, 2])).not.toBe(stateHash([2, 1]));
  });
});

describe("model math", () => {
  it.each([
    [4, 128, 1_377_408],
    [6, 192, 3_443_136],
    [8, 256, 7_475_456],
    [10, 384, 19_275_648],
    [12, 512, 40_645_120],
  ])("paramCount(%i, %i) matches the Python trainer", (nLayer, dModel, expected) => {
    expect(paramCount(nLayer, dModel)).toBe(expected);
  });

  it("counts training compute as 6 x params x tokens", () => {
    expect(trainingFlops(1377408, 12582912)).toBe(6 * 1377408 * 12582912);
  });
});

describe("calendar", () => {
  it.each([
    [0, "2027-01-01"],
    [58, "2027-02-28"],
    [365, "2028-01-01"],
    [424, "2028-02-29"],
  ])("day %i is %s", (day, date) => {
    expect(dateOfDay(day)).toBe(date);
  });

  it("agrees with the platform calendar for 60 years", () => {
    const start = Date.UTC(2027, 0, 1);
    for (let day = 0; day < 365 * 60; day += 7) {
      expect(dateOfDay(day)).toBe(new Date(start + day * 86_400_000).toISOString().slice(0, 10));
    }
  });
});

describe("balance", () => {
  it("prices the first model at about a sixth of the founding cash, over 13 game days", () => {
    const b = BALANCE_V2;
    const tiny = b.brainSizes.find((s) => s.id === b.defaultPlan.brainSize);
    const length = b.lengths.find((l) => l.id === b.defaultPlan.length);
    if (!tiny || !length) throw new Error("default plan names a missing option");
    const pflop = trainingFlops(paramCount(tiny.nLayer, tiny.dModel), length.tokens) / 1e15;
    const days = Math.ceil(pflop / b.cloudPflopsPerDay);
    const cost = Math.round(pflop * b.cloudPricePerPflop);
    expect(days).toBe(13);
    expect(cost).toBe(41_596);
    expect(days * b.tickMsAt1x).toBe(104_000);
    expect(b.foundingCash / cost).toBeGreaterThan(5);
    expect(b.foundingCash / cost).toBeLessThan(7);
  });

  it.each([BALANCE_V2, BALANCE_TEST])(
    "$version has a valid default plan built from its own options",
    (b) => {
      const plan = TrainingPlan.parse(b.defaultPlan);
      expect(b.brainSizes.map((s) => s.id)).toContain(plan.brainSize);
      expect(b.attentionSpans).toContain(plan.attentionSpan);
      expect(b.lengths.map((l) => l.id)).toContain(plan.length);
      expect(Object.values(plan.mix).every((w) => w === 12)).toBe(true);
    },
  );

  it("trains the test model in 30 game days, 6 s at 1x", () => {
    const b = BALANCE_TEST;
    const pflop = trainingFlops(paramCount(2, 64), 65_536) / 1e15;
    expect(Math.ceil(pflop / b.cloudPflopsPerDay)).toBe(30);
    expect(30 * b.tickMsAt1x).toBe(6000);
  });

  it("looks balances up by version", () => {
    expect(getBalance("balance-v2")).toBe(BALANCE_V2);
    expect(getBalance("balance-test")).toBe(BALANCE_TEST);
    expect(BALANCE_TEST.tickMsAt1x).toBe(200);
    expect(BALANCE_TEST.brainSizes).toEqual([{ id: "tiny", nLayer: 2, dModel: 64 }]);
  });
});

describe("balance v2 economy", () => {
  it("looks the economy balances up by version", () => {
    expect(getBalance("balance-v2")).toBe(BALANCE_V2);
    expect(getBalance("balance-test")).toBe(BALANCE_TEST);
  });

  it.each([BALANCE_V2, BALANCE_TEST])(
    "$version mixes groups to 1 and weighs only bench categories",
    (b) => {
      for (const region of REGION_IDS) {
        const mix = b.economy.groupMix[region];
        expect(GROUP_IDS.reduce((sum, g) => sum + mix[g], 0)).toBeCloseTo(1, 9);
      }
      for (const g of GROUP_IDS) {
        for (const category of Object.keys(b.economy.groups[g].weights)) {
          expect(BENCH_CATEGORIES).toContain(category);
        }
      }
    },
  );

  it("starts adoption near 10% and tends to 60%", () => {
    expect(adoptionShare(BALANCE_V2.economy, 0)).toBeCloseTo(0.1, 2);
    expect(adoptionShare(BALANCE_V2.economy, 365 * 40)).toBeCloseTo(0.6, 2);
    expect(adoptionShare(BALANCE_V2.economy, 365 * 4)).toBeGreaterThan(
      adoptionShare(BALANCE_V2.economy, 365),
    );
  });

  it("keeps default prices inside their bounds", () => {
    for (const plan of ["personal", "business"] as const) {
      const p = BALANCE_V2.economy.prices[plan];
      expect(p.default).toBeGreaterThanOrEqual(p.min);
      expect(p.default).toBeLessThanOrEqual(p.max);
    }
    expect(BALANCE_V2.economy.prices.personal).toEqual({ default: 10, min: 1, max: 200 });
    expect(BALANCE_V2.economy.prices.business).toEqual({ default: 30, min: 5, max: 500 });
  });
});
