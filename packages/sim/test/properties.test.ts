import { WorldView } from "@airace/protocol";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { BALANCE_V1, projectView, stateHash } from "../src/index.ts";
import { arbInputs, newWorld, runInputs } from "./arbitraries.ts";

const RUNS = { numRuns: 200 };

describe("sim properties", () => {
  it("(a) privacy: no viewer sees another company's cash, runs, prompts or unshared versions", () => {
    fc.assert(
      fc.property(arbInputs(100), (inputs) => {
        const s = runInputs(newWorld(), inputs);
        for (const viewer of Object.values(s.members)) {
          const view = WorldView.parse(projectView(s, viewer.userId));
          const text = JSON.stringify(view);
          // Only the viewer's own company carries a cash figure.
          expect(text.split('"cash":').length - 1).toBe(viewer.companyId ? 1 : 0);
          for (const company of Object.values(s.companies)) {
            if (company.id === viewer.companyId) continue;
            for (const run of Object.values(company.runs)) {
              expect(text).not.toContain(run.id);
              if (run.plan.probePrompt) expect(text).not.toContain(run.plan.probePrompt);
            }
            for (const version of Object.values(company.versions)) {
              if (!version.published) expect(text).not.toContain(version.id);
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

  it("(d) money: cash is founding cash minus charged runs plus refunds, never negative", () => {
    fc.assert(
      fc.property(arbInputs(100), (inputs) => {
        const s = runInputs(newWorld(), inputs);
        for (const company of Object.values(s.companies)) {
          const charged = Object.values(company.runs)
            .filter((r) => r.status !== "failed")
            .reduce((sum, r) => sum + r.preview.cost, 0);
          expect(company.cash).toBe(BALANCE_V1.foundingCash - charged);
          expect(company.cash).toBeGreaterThanOrEqual(0);
        }
      }),
      RUNS,
    );
  });

  it("explores worlds where training happens", () => {
    // Guards the generator: the properties above mean little if no run ever completes.
    const stats = { runs: 0, versions: 0, failed: 0 };
    fc.assert(
      fc.property(arbInputs(100), (inputs) => {
        const s = runInputs(newWorld(), inputs);
        for (const c of Object.values(s.companies)) {
          stats.runs += Object.keys(c.runs).length;
          stats.versions += Object.keys(c.versions).length;
          stats.failed += Object.values(c.runs).filter((r) => r.status === "failed").length;
        }
      }),
      { numRuns: 200, seed: 1 },
    );
    expect(stats.runs).toBeGreaterThan(50);
    expect(stats.versions).toBeGreaterThan(10);
    expect(stats.failed).toBeGreaterThan(5);
  });
});
