import { GROUP_IDS, PLAN_IDS, REGION_IDS, WorldView } from "@airace/protocol";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { adoptable, BALANCE_V2, customersOf, projectView, stateHash } from "../src/index.ts";
import { arbInputs, newWorld, runInputs } from "./arbitraries.ts";

const RUNS = { numRuns: 200 };

// Each property runs 200 worlds; under the full parallel suite that can take well over 5 s.
describe("sim properties", { timeout: 30_000 }, () => {
  it("(a) privacy: no viewer sees another company's money, satisfaction, training or unreleased AIs", () => {
    const count = (text: string, key: string) => text.split(`"${key}":`).length - 1;
    fc.assert(
      fc.property(arbInputs(100), (inputs) => {
        const s = runInputs(newWorld(), inputs);
        for (const viewer of Object.values(s.members)) {
          const view = WorldView.parse(projectView(s, viewer.userId));
          const text = JSON.stringify(view);
          // Private fields appear once, in the viewer's own company, or not at all.
          const own = viewer.companyId ? s.companies[viewer.companyId] : undefined;
          for (const key of [
            "ledger",
            "satisfaction",
            "plans",
            "fame",
            "history",
            "byRegion",
            "campaigns",
            "compute",
            "dataCenters",
          ]) {
            expect(count(text, key), key).toBe(own ? 1 : 0);
          }
          // Cash shows in the viewer's own company and in each day of its history.
          expect(count(text, "cash")).toBe(own ? 1 + own.history.length : 0);
          expect(count(text, "hype")).toBe(own ? Object.keys(own.products).length : 0);
          expect(count(text, "awareness")).toBe(0);
          for (const company of Object.values(s.companies)) {
            if (company.id === viewer.companyId) continue;
            for (const run of Object.values(company.runs)) {
              expect(text).not.toContain(run.id);
              if (run.plan.probePrompt) expect(text).not.toContain(run.plan.probePrompt);
            }
            // An AI is public once shared, or while a live product runs on it.
            const live = new Set(
              Object.values(company.products)
                .filter((p) => p.status === "live")
                .map((p) => p.versionId),
            );
            for (const version of Object.values(company.versions)) {
              if (!version.published && !live.has(version.id)) {
                expect(text).not.toContain(version.id);
              }
            }
          }
        }
      }),
      RUNS,
    );
  });

  it("(b) determinism: the same inputs give the same state", () => {
    fc.assert(
      fc.property(arbInputs(100), (inputs) => {
        expect(stateHash(runInputs(newWorld(), inputs))).toBe(
          stateHash(runInputs(newWorld(), inputs)),
        );
      }),
      RUNS,
    );
  });

  it("(c) replay: a snapshot taken anywhere, stored as JSON, continues to the same state", () => {
    fc.assert(
      fc.property(arbInputs(100), fc.nat(), (inputs, cut) => {
        const k = inputs.length === 0 ? 0 : cut % (inputs.length + 1);
        const snapshot = JSON.parse(JSON.stringify(runInputs(newWorld(), inputs.slice(0, k))));
        const resumed = runInputs(snapshot, inputs.slice(k));
        expect(stateHash(resumed)).toBe(stateHash(runInputs(newWorld(), inputs)));
      }),
      RUNS,
    );
  });

  // Cash may fall below zero (serving costs); rescue at a month end is Phase 2.
  it("(d) money: cash is founding cash plus income less costs plus financing, each booked once", () => {
    const total = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);
    fc.assert(
      fc.property(arbInputs(100), (inputs) => {
        const s = runInputs(newWorld(), inputs);
        for (const company of Object.values(s.companies)) {
          const months = [...company.ledger.months, company.ledger.current];
          const net = months.reduce(
            (sum, m) =>
              sum +
              total(m.income) -
              total(m.costs) +
              m.financing.loans +
              m.financing.investors -
              m.financing.repaid,
            0,
          );
          expect(Number.isInteger(company.cash)).toBe(true);
          expect(company.cash).toBe(BALANCE_V2.foundingCash + net);
          // Training money: every run not refunded, once.
          const charged = Object.values(company.runs)
            .filter((r) => r.status !== "failed")
            .reduce((sum, r) => sum + r.preview.cost, 0);
          const booked = months.reduce((sum, m) => sum + m.costs.training + m.costs.dataChecks, 0);
          expect(booked).toBe(charged);
        }
      }),
      RUNS,
    );
  });

  it("(e) customers are never negative and never more than the people willing", () => {
    fc.assert(
      fc.property(arbInputs(100), (inputs) => {
        const s = runInputs(newWorld(), inputs);
        const e = BALANCE_V2.economy;
        for (const region of REGION_IDS) {
          for (const group of GROUP_IDS) {
            let people = 0;
            for (const id of Object.keys(s.companies)) {
              for (const plan of PLAN_IDS) {
                const n = customersOf(s.market, id, region, group, plan);
                expect(n).toBeGreaterThanOrEqual(0);
                people += plan === "business" ? n * e.groups[group].seatsPerAccount : n;
              }
            }
            expect(people).toBeLessThanOrEqual(adoptable(e, s.day, region, group) * (1 + 1e-9));
          }
        }
      }),
      RUNS,
    );
  });

  it("explores worlds where training, launches and customer posts happen", () => {
    // Guards the generator: the properties above mean little if nothing ever happens.
    const stats = { runs: 0, versions: 0, failed: 0, launches: 0, posts: 0 };
    fc.assert(
      fc.property(arbInputs(100), (inputs) => {
        const s = runInputs(newWorld(), inputs);
        for (const c of Object.values(s.companies)) {
          stats.runs += Object.keys(c.runs).length;
          stats.versions += Object.keys(c.versions).length;
          stats.failed += Object.values(c.runs).filter((r) => r.status === "failed").length;
          stats.launches += Object.values(c.products).filter(
            (p) => p.status !== "countdown",
          ).length;
        }
        stats.posts += s.feed.filter((i) => i.kind === "post").length;
      }),
      { numRuns: 200, seed: 1 },
    );
    expect(stats.runs).toBeGreaterThan(50);
    expect(stats.versions).toBeGreaterThan(10);
    expect(stats.failed).toBeGreaterThan(5);
    expect(stats.launches).toBeGreaterThan(0);
    expect(stats.posts).toBeGreaterThan(0);
  });
});
