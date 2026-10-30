import { describe, expect, it } from "vitest";
import { step, upgradeState } from "../src/index.ts";
import { runningWorld, TICK, ticks } from "./pack.ts";

/** A sim-3 state: today's state without the fields sim-4 added. */
function sim3() {
  const s = structuredClone(ticks(runningWorld(), 3)) as unknown as Record<string, unknown> & {
    companies: Record<string, Record<string, unknown> & { versions: Record<string, object> }>;
  };
  delete s.rivals;
  delete s.leaders;
  delete s.lastRankingNews;
  for (const c of Object.values(s.companies)) {
    delete c.rival;
    delete c.startCash;
    delete c.lastRanks;
    for (const v of Object.values(c.versions)) delete (v as { packKey?: unknown }).packKey;
  }
  return s;
}

describe("upgrading saved worlds", () => {
  it("brings a sim-3 world to sim-4 and keeps it running", () => {
    const up = upgradeState(sim3(), "sim-3");
    expect(up.rivals).toBeNull();
    expect(up.companies.c1?.rival).toBeNull();
    expect(up.companies.c1?.startCash).toBe(250_000);
    expect(up.companies.c1?.lastRanks).toBeNull();
    expect(up.leaders).toEqual({});
    expect(up.lastRankingNews).toEqual({});
    const next = step(up, TICK);
    expect(next.ok).toBe(true);
  });
});
