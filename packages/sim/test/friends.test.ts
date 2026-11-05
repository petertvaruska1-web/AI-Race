import { describe, expect, it } from "vitest";
import {
  BALANCE_V2,
  type CompanyState,
  createWorld,
  foundingCashOn,
  projectView,
  step,
  upgradeState,
  type VersionState,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys, TICK, ticks } from "./pack.ts";

const LOGO = { shape: "circle", symbol: "spark", initials: "BE" } as const;
const PALETTE = { primary: "#3a86ff", secondary: "#ffbe0b" };

/** `s` with u2 joined and founding company c2. */
function withFriend(s0: WorldState): WorldState {
  let s = ok(step(s0, sys({ type: "memberJoined", userId: "u2", displayName: "Ben" })));
  s = ok(
    step(s, cmd({ type: "createCompany", name: "Beta", logo: LOGO, palette: PALETTE }, "c2", "u2")),
  );
  return s;
}

const c = (s: WorldState, id: string) => s.companies[id] as CompanyState;
const offline = (s: WorldState, userId: string) =>
  ok(step(s, sys({ type: "presence", userId, online: false })));

describe("Fair mode", () => {
  it("is set when the world is created, off by default", () => {
    const base = {
      worldId: "w1",
      name: "Friends",
      hostUserId: "u1",
      hostName: "Ana",
      seed: 1,
      balance: "balance-v2" as const,
      inviteCode: "INV",
    };
    expect(createWorld(base).fairMode).toBe(false);
    const fair = createWorld({ ...base, fairMode: true });
    expect(fair.fairMode).toBe(true);
    expect(projectView(fair, "u1").world.fairMode).toBe(true);
  });
});

describe("the founder grant", () => {
  it("gives a late founder more cash the older the world, up to double", () => {
    const f = BALANCE_V2.foundingCash;
    expect(foundingCashOn(BALANCE_V2, 0)).toBe(f);
    expect(foundingCashOn(BALANCE_V2, 30)).toBe(f + 60_000);
    expect(foundingCashOn(BALANCE_V2, 200)).toBe(2 * f);
    const s = withFriend(ticks(runningWorld(), 30));
    expect(c(s, "c2").cash).toBe(f + 60_000);
    expect(c(s, "c2").startCash).toBe(f + 60_000);
  });
});

describe("autopilot", () => {
  function campaigning(): WorldState {
    const s = ticks(runningWorld(), 1);
    c(s, "c1").campaigns.ad1 = {
      id: "ad1",
      region: "all",
      group: "all",
      budgetPerDay: 1_000,
      startDay: s.day,
      endDay: s.day + 30,
      spent: 0,
    };
    return s;
  }

  it("pauses an absent owner's ads on a day its cash is negative", () => {
    let s = offline(campaigning(), "u1");
    c(s, "c1").cash = -10;
    const spent = c(s, "c1").campaigns.ad1?.spent ?? 0;
    s = ok(step(s, TICK));
    expect(c(s, "c1").campaigns.ad1?.spent).toBe(spent);
    s = ok(step(s, sys({ type: "presence", userId: "u1", online: true })));
    s = ok(step(s, TICK));
    expect(c(s, "c1").campaigns.ad1?.spent).toBe(spent + 1_000);
  });

  it("keeps paying for the ads of an absent owner with cash", () => {
    let s = offline(campaigning(), "u1");
    const spent = c(s, "c1").campaigns.ad1?.spent ?? 0;
    s = ok(step(s, TICK));
    expect(c(s, "c1").campaigns.ad1?.spent).toBe(spent + 1_000);
  });

  it("holds an absent owner's rescue offer until they are back", () => {
    let s = ticks(runningWorld(), 1);
    c(s, "c1").rescue = { offeredDay: s.day, deadlineDay: s.day + 1 };
    c(s, "c1").cash = -5_000;
    s = offline(s, "u1");
    s = ticks(s, 5);
    expect(c(s, "c1").loan).toBeNull();
    expect(c(s, "c1").rescue).not.toBeNull();
    s = ok(step(s, sys({ type: "presence", userId: "u1", online: true })));
    s = ticks(s, 2);
    expect(c(s, "c1").loan).not.toBeNull();
  });
});

describe("upgrading a sim-6 world", () => {
  it("comes up without Fair mode and keeps playing", () => {
    const old = structuredClone(runningWorld()) as unknown as Record<string, unknown>;
    delete old.fairMode;
    const s = upgradeState(old, "sim-6");
    expect(s.fairMode).toBe(false);
    expect(step(s, TICK).ok).toBe(true);
  });
});

describe("sharing an AI with friends", () => {
  function withVersion(s0: WorldState): WorldState {
    const s = structuredClone(s0);
    (s.companies.c1 as CompanyState).versions.v1 = {
      id: "v1",
      label: "Spark 1",
      runId: "r-v1",
      parentId: null,
      nLayer: 2,
      dModel: 64,
      ctxLen: 64,
      createdDay: 0,
      trainedTokens: 1,
      scores: null,
      overall: null,
      personalityWords: [],
      traits: null,
      published: false,
      packKey: null,
    } as VersionState;
    return s;
  }

  it("shares and unshares an AI; friends see only the shared ones", () => {
    let s = withFriend(withVersion(runningWorld()));
    expect(projectView(s, "u2").companies.find((x) => x.id === "c1")?.publishedVersions).toEqual(
      [],
    );
    s = ok(step(s, cmd({ type: "publishVersion", versionId: "v1", published: true })));
    const shared = projectView(s, "u2").companies.find((x) => x.id === "c1")?.publishedVersions;
    expect(shared?.map((v) => v.id)).toEqual(["v1"]);
    s = ok(step(s, cmd({ type: "publishVersion", versionId: "v1", published: false })));
    expect(c(s, "c1").versions.v1?.published).toBe(false);
  });

  it("refuses to share someone else's AI", () => {
    const s = withFriend(withVersion(runningWorld()));
    expect(
      rejected(
        step(s, cmd({ type: "publishVersion", versionId: "v1", published: true }, undefined, "u2")),
      ),
    ).toBe("That AI isn't one of yours.");
  });
});
