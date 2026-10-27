import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Catalog, DATASET_IDS, DATASET_TITLES, PREP_KEYS, prepKey } from "../src/index.ts";

const sample = () =>
  JSON.parse(readFileSync(new URL("../fixtures/catalog.sample.json", import.meta.url), "utf8"));

describe("Catalog", () => {
  it("parses the sample export", () => {
    const c = Catalog.parse(sample());
    expect(c.datasets.map((d) => d.id)).toEqual([...DATASET_IDS]);
    expect(c.datasets.map((d) => d.title)).toEqual(DATASET_IDS.map((id) => DATASET_TITLES[id]));
  });

  it("knows the 12 ways to prepare a dataset", () => {
    expect(PREP_KEYS).toHaveLength(12);
    expect(new Set(PREP_KEYS).size).toBe(12);
    expect(prepKey("thorough", true, false)).toBe("thorough|1|0");
    expect(PREP_KEYS).toContain(prepKey("light", false, true));
  });

  it("refuses a dataset missing one of its preparation pools", () => {
    const c = sample();
    delete c.datasets[0].pools["light|0|0"];
    expect(Object.keys(c.datasets[0].pools)).toHaveLength(11);
    expect(Catalog.safeParse(c).success).toBe(false);
  });

  it("refuses a missing or repeated dataset and over-long samples", () => {
    const missing = sample();
    missing.datasets.pop();
    expect(Catalog.safeParse(missing).success).toBe(false);
    const repeated = sample();
    repeated.datasets[1] = { ...repeated.datasets[0] };
    expect(Catalog.safeParse(repeated).success).toBe(false);
    const long = sample();
    long.datasets[0].samples = ["x".repeat(601)];
    expect(Catalog.safeParse(long).success).toBe(false);
  });
});
