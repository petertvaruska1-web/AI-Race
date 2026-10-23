import { type Command, WorldView } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  BALANCE_V1,
  type Input,
  projectView,
  type StepResult,
  step,
  type WorldState,
} from "../src/index.ts";
import { newWorld } from "./arbitraries.ts";

function ok(r: StepResult): WorldState {
  if (!r.ok) throw new Error(`rejected: ${r.reason}`);
  return r.state;
}

const cmd = (userId: string, c: unknown, newId?: string): Input => ({
  kind: "command",
  userId,
  cmd: c as Command,
  ...(newId === undefined ? {} : { newId }),
});

/** u1 owns CO_A with a trained v1 (scored) and a second run in training; u2 owns CO_B. */
function world(): WorldState {
  let s = newWorld();
  s = ok(
    step(s, { kind: "system", event: { type: "memberJoined", userId: "u2", displayName: "Ben" } }),
  );
  s = ok(
    step(
      s,
      cmd(
        "u1",
        {
          type: "createCompany",
          name: "Acme",
          logo: { shape: "circle", symbol: "spark", initials: "AC" },
          palette: { primary: "#4f8cff", secondary: "#ffb020" },
        },
        "CO_A",
      ),
    ),
  );
  s = ok(
    step(
      s,
      cmd(
        "u2",
        {
          type: "createCompany",
          name: "Beta",
          logo: { shape: "shield", symbol: null, initials: "B" },
          palette: { primary: "#ff7a45", secondary: "#ffd166" },
        },
        "CO_B",
      ),
    ),
  );
  s = ok(step(s, cmd("u1", { type: "startWorld" })));
  const plan = {
    ...BALANCE_V1.defaultPlan,
    mix: { ...BALANCE_V1.defaultPlan.mix },
    probePrompt: "SECRET_PROMPT_A",
  };
  s = ok(step(s, cmd("u1", { type: "startTraining", plan }, "RUN_A1")));
  s = ok(
    step(s, {
      kind: "system",
      event: {
        type: "trainingCompleted",
        runId: "RUN_A1",
        versionId: "VER_A1",
        result: { steps: 1, finalLoss: 4, heldoutLosses: {}, unstable: false },
      },
    }),
  );
  for (let i = 0; i < 13; i++) s = ok(step(s, { kind: "tick" }));
  s = ok(
    step(s, {
      kind: "system",
      event: {
        type: "evalCompleted",
        versionId: "VER_A1",
        scores: { language: 40 },
        overall: 21,
        personalityWords: ["curious"],
      },
    }),
  );
  s = ok(
    step(
      s,
      cmd("u1", { type: "startTraining", plan: { ...plan, parentVersionId: "VER_A1" } }, "RUN_A2"),
    ),
  );
  s = ok(
    step(s, {
      kind: "system",
      event: {
        type: "trainingCompleted",
        runId: "RUN_A2",
        versionId: "VER_A2",
        result: { steps: 1, finalLoss: 3.9, heldoutLosses: {}, unstable: false },
      },
    }),
  );
  return s;
}

describe("projectView", () => {
  it("shows the owner their cash, runs and versions", () => {
    const view = WorldView.parse(projectView(world(), "u1"));
    expect(view.me.company?.id).toBe("CO_A");
    expect(view.me.company?.cash).toBe(BALANCE_V1.foundingCash - 2 * 41_596);
    expect(view.me.company?.runs.map((r) => [r.id, r.status])).toEqual([
      ["RUN_A1", "completed"],
      ["RUN_A2", "training"],
    ]);
    expect(view.me.company?.versions.map((v) => v.label)).toEqual(["v1"]);
    expect(view.me.company?.versions[0]?.shape).toEqual({ nLayer: 4, dModel: 128, ctxLen: 256 });
    expect(view.world).toMatchObject({
      day: 13,
      date: "2027-01-14",
      inviteCode: "INVITE-42",
      effectiveHostUserId: "u1",
    });
  });

  it("keeps an early result hidden until its game day, even from the owner", () => {
    const text = JSON.stringify(projectView(world(), "u1"));
    expect(text).not.toContain("VER_A2");
    expect(text).not.toContain("3.9");
  });

  it("shows another company only its public identity and shared versions", () => {
    const s = world();
    const view = WorldView.parse(projectView(s, "u2"));
    const acme = view.companies.find((c) => c.id === "CO_A");
    expect(Object.keys(acme ?? {}).sort()).toEqual([
      "id",
      "logo",
      "name",
      "ownerUserId",
      "palette",
      "publishedVersions",
    ]);
    expect(acme?.publishedVersions).toEqual([]);
    const text = JSON.stringify(view);
    for (const secret of [
      "RUN_A1",
      "RUN_A2",
      "VER_A1",
      "SECRET_PROMPT_A",
      String(s.companies.CO_A?.cash),
    ]) {
      expect(text).not.toContain(secret);
    }

    const shared = structuredClone(s);
    const v1 = shared.companies.CO_A?.versions.VER_A1;
    if (!v1) throw new Error("missing version");
    v1.published = true;
    const published = WorldView.parse(projectView(shared, "u2")).companies.find(
      (c) => c.id === "CO_A",
    );
    expect(published?.publishedVersions).toEqual([
      {
        id: "VER_A1",
        label: "v1",
        createdDay: 13,
        scores: { language: 40 },
        overall: 21,
        personalityWords: ["curious"],
      },
    ]);
  });

  it("lists members in joining order and refuses a non-member", () => {
    const s = world();
    expect(projectView(s, "u2").members.map((m) => m.userId)).toEqual(["u1", "u2"]);
    expect(projectView(s, "u2").me).toMatchObject({ userId: "u2", company: { id: "CO_B" } });
    expect(() => projectView(s, "u9")).toThrow();
  });

  it("shares no objects with the state", () => {
    const s = world();
    const view = projectView(s, "u1");
    const company = view.me.company;
    if (!company?.runs[0]) throw new Error("missing run");
    company.runs[0].plan.mix.web = 99;
    expect(s.companies.CO_A?.runs.RUN_A1?.plan.mix.web).toBe(12);
  });
});
