import type { Command } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  BALANCE_V1,
  contrastRatio,
  createWorld,
  effectiveHost,
  type Input,
  PALETTE_PRESETS,
  paletteProblem,
  type StepResult,
  step,
  type WorldState,
} from "../src/index.ts";

const LOGO = { shape: "circle", symbol: "spark", initials: "AC" } as const;
const PALETTE = { primary: "#4f8cff", secondary: "#ffb020" };

function ok(r: StepResult): WorldState {
  if (!r.ok) throw new Error(`rejected: ${r.reason}`);
  return r.state;
}

function rejected(r: StepResult): string {
  if (r.ok) throw new Error("expected a rejection");
  return r.reason;
}

function world(): WorldState {
  return createWorld({
    worldId: "w1",
    name: "Friday night",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 1,
    balance: "balance-v1",
    inviteCode: "ABCD-1234",
  });
}

const cmd = (userId: string, c: unknown, newId?: string): Input => ({
  kind: "command",
  userId,
  cmd: c as Command,
  ...(newId === undefined ? {} : { newId }),
});

const found = (userId: string, name: string, newId: string, palette = PALETTE): Input =>
  cmd(userId, { type: "createCompany", name, logo: LOGO, palette }, newId);

const join = (userId: string, displayName: string): Input => ({
  kind: "system",
  event: { type: "memberJoined", userId, displayName },
});

const presence = (userId: string, online: boolean): Input => ({
  kind: "system",
  event: { type: "presence", userId, online },
});

const TICK: Input = { kind: "tick" };

function running(): WorldState {
  let s = ok(step(world(), found("u1", "Acme", "c1")));
  s = ok(step(s, cmd("u1", { type: "startWorld" })));
  return s;
}

describe("createWorld", () => {
  it("starts in setup on day 0 with the host as the only member", () => {
    const s = world();
    expect(s.phase).toBe("setup");
    expect(s.day).toBe(0);
    expect(s.seq).toBe(0);
    expect(s.speed).toBe(1);
    expect(Object.keys(s.members)).toEqual(["u1"]);
    expect(s.members.u1?.companyId).toBeNull();
  });
});

describe("createCompany", () => {
  it("founds a company with the founding cash, using the input's id", () => {
    const s = ok(step(world(), found("u1", "  Acme\u0007 ", "c1")));
    expect(s.companies.c1).toMatchObject({
      id: "c1",
      ownerUserId: "u1",
      name: "Acme",
      cash: BALANCE_V1.foundingCash,
      versionSeq: 0,
    });
    expect(s.members.u1?.companyId).toBe("c1");
    expect(s.seq).toBe(1);
  });

  it("allows one company per member", () => {
    const s = ok(step(world(), found("u1", "Acme", "c1")));
    expect(rejected(step(s, found("u1", "Beta", "c2")))).toMatch(/already have a company/i);
  });

  it("refuses a name another company uses, in any case", () => {
    let s = ok(step(world(), join("u2", "Ben")));
    s = ok(step(s, found("u1", "Acme", "c1")));
    expect(rejected(step(s, found("u2", "ACME", "c2")))).toMatch(/name/i);
    expect(ok(step(s, found("u2", "Acme Two", "c2"))).companies.c2?.name).toBe("Acme Two");
  });

  it("refuses a palette that would not show on the dark dashboard", () => {
    const dark = { primary: "#101010", secondary: "#ffb020" };
    expect(rejected(step(world(), found("u1", "Acme", "c1", dark)))).toMatch(/too dark/i);
  });

  it("needs the server-made id and a valid command", () => {
    expect(
      rejected(
        step(
          world(),
          cmd("u1", { type: "createCompany", name: "Acme", logo: LOGO, palette: PALETTE }),
        ),
      ),
    ).toMatch(/id/i);
    expect(rejected(step(world(), found("u1", "A", "c1")))).toMatch(/2.32/);
  });

  it("refuses commands from someone who is not a member", () => {
    expect(rejected(step(world(), found("u9", "Acme", "c1")))).toMatch(/member/i);
  });

  it("leaves the given state unchanged, accepted or not", () => {
    const s = world();
    const before = structuredClone(s);
    step(s, found("u1", "Acme", "c1"));
    step(s, found("u1", "A", "c1"));
    expect(s).toEqual(before);
  });
});

describe("startWorld", () => {
  it("is for the host, in setup, once the host has a company", () => {
    let s = ok(step(world(), join("u2", "Ben")));
    expect(rejected(step(s, cmd("u1", { type: "startWorld" })))).toMatch(/company/i);
    s = ok(step(s, found("u1", "Acme", "c1")));
    s = ok(step(s, found("u2", "Beta", "c2")));
    expect(rejected(step(s, cmd("u2", { type: "startWorld" })))).toMatch(/host/i);
    s = ok(step(s, cmd("u1", { type: "startWorld" })));
    expect(s.phase).toBe("running");
    expect(rejected(step(s, cmd("u1", { type: "startWorld" })))).toMatch(/already/i);
  });
});

describe("clock", () => {
  it("does not tick before the world starts or while paused", () => {
    expect(rejected(step(world(), TICK))).toMatch(/not running/i);
    const paused = ok(step(running(), cmd("u1", { type: "setPaused", paused: true })));
    expect(rejected(step(paused, TICK))).toMatch(/not running/i);
    const resumed = ok(step(paused, cmd("u1", { type: "setPaused", paused: false })));
    expect(ok(step(resumed, TICK)).day).toBe(1);
  });

  it("advances one day per tick and counts every accepted input", () => {
    let s = running();
    const seq = s.seq;
    s = ok(step(s, TICK));
    s = ok(step(s, TICK));
    expect(s.day).toBe(2);
    expect(s.seq).toBe(seq + 2);
  });

  it("takes only the allowed speeds, from the host", () => {
    let s = ok(step(running(), join("u2", "Ben")));
    expect(rejected(step(s, cmd("u1", { type: "setSpeed", speed: 3 })))).toBeTruthy();
    expect(rejected(step(s, cmd("u2", { type: "setSpeed", speed: 4 })))).toMatch(/host/i);
    s = ok(step(s, cmd("u1", { type: "setSpeed", speed: 4 })));
    expect(s.speed).toBe(4);
  });
});

describe("members and presence", () => {
  it("hands hosting to the earliest-joined online member while the host is away", () => {
    let s = ok(step(running(), join("u2", "Ben")));
    s = ok(step(s, join("u3", "Cy")));
    expect(effectiveHost(s)).toBe("u1");
    s = ok(step(s, presence("u1", false)));
    expect(effectiveHost(s)).toBe("u2");
    expect(s.members.u1?.online).toBe(false);
    expect(ok(step(s, cmd("u2", { type: "setPaused", paused: true }))).paused).toBe(true);
    s = ok(step(s, presence("u2", false)));
    s = ok(step(s, presence("u3", false)));
    expect(effectiveHost(s)).toBeNull();
    s = ok(step(s, presence("u1", true)));
    expect(effectiveHost(s)).toBe("u1");
  });

  it("refuses a second join, an unknown member's presence and a full world", () => {
    let s = world();
    expect(rejected(step(s, join("u1", "Ana")))).toMatch(/already/i);
    expect(rejected(step(s, presence("u9", true)))).toMatch(/member/i);
    for (let i = 2; i <= BALANCE_V1.maxMembers; i++) s = ok(step(s, join(`u${i}`, `P${i}`)));
    expect(rejected(step(s, join("u99", "Late")))).toMatch(/full/i);
  });
});

describe("palettes", () => {
  it("measures contrast the WCAG way", () => {
    expect(contrastRatio("#000000", "#ffffff")).toBeCloseTo(21, 5);
    expect(contrastRatio("#4f8cff", "#4F8CFF")).toBeCloseTo(1, 5);
  });

  it("needs two different colors that both stand out", () => {
    expect(paletteProblem({ primary: "#4f8cff", secondary: "#4F8CFF" })).toMatch(/different/i);
    expect(paletteProblem({ primary: "#4f8cff", secondary: "#1a1a1a" })).toMatch(/too dark/i);
    expect(paletteProblem(PALETTE)).toBeNull();
  });

  it("offers 12 presets that all pass", () => {
    expect(PALETTE_PRESETS).toHaveLength(12);
    expect(new Set(PALETTE_PRESETS.map((p) => p.id)).size).toBe(12);
    for (const p of PALETTE_PRESETS) expect(paletteProblem(p.palette), p.id).toBeNull();
  });
});
