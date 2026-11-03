import { describe, expect, it } from "vitest";
import { type CompanyState, getBalance, projectView, step, type WorldState } from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys, ticks } from "./pack.ts";

const B = getBalance("balance-test").economy.showcases;

/** c1 sells "Spark" (v1), live. */
function live(): WorldState {
  const s = structuredClone(runningWorld("balance-test"));
  const c1 = s.companies.c1 as CompanyState;
  c1.versions.v1 = {
    id: "v1",
    label: "v1",
    runId: "r1",
    parentId: null,
    nLayer: 2,
    dModel: 64,
    ctxLen: 64,
    createdDay: s.day,
    trainedTokens: 1,
    scores: { creativity: 20 },
    overall: 30,
    personalityWords: [],
    traits: null,
    published: false,
    packKey: null,
  };
  const launched = ok(
    step(
      s,
      cmd(
        {
          type: "launchProduct",
          versionId: "v1",
          name: "Spark",
          plans: ["free"],
          countdownDays: 3,
          keynote: null,
        },
        "p1",
      ),
    ),
  );
  return ticks(launched, 4);
}

const start = (s: WorldState, over: Record<string, unknown> = {}, id = "sh1") =>
  step(
    s,
    cmd(
      {
        type: "startShowcase",
        game: "bridge",
        difficulty: "easy",
        versionId: "v1",
        settings: {},
        ...over,
      },
      id,
    ),
  );
const resolve = (s: WorldState, e: Record<string, unknown>) =>
  step(s, sys({ type: "showcaseResolved", showcaseId: "sh1", ...e }));
const c1 = (s: WorldState) => s.companies.c1 as CompanyState;

describe("showcases", () => {
  it("charge the fee and remember the game they will play", () => {
    const s = live();
    const after = ok(start(s));
    expect(c1(after).cash).toBe(c1(s).cash - B.fee);
    expect(c1(after).ledger.current.costs.events).toBe(B.fee);
    expect(c1(after).showcases[0]).toMatchObject({
      id: "sh1",
      game: "bridge",
      status: "running",
      productName: "Spark",
    });
  });

  it("need a live AI, one at a time, a pause between, and the money", () => {
    const s = live();
    expect(rejected(start(s, { versionId: "nope" }))).toMatch(/on sale/);
    const once = ok(start(s));
    expect(rejected(start(once, {}, "sh2"))).toMatch(/still going/);
    const done = ok(resolve(once, { score: 6, maxScore: 12 }));
    expect(rejected(start(done, {}, "sh2"))).toMatch(/Try again on/);
    const later = ticks(done, B.cooldownDays + 1);
    expect(start(later, {}, "sh2").ok).toBe(true);
    const poor = structuredClone(s);
    c1(poor).cash = B.fee - 1;
    expect(rejected(start(poor))).toMatch(/costs/);
  });

  it("a great run wins fans and good news", () => {
    const s = ok(start(live()));
    const followers = c1(s).fame.followers;
    const done = ok(resolve(s, { score: 12, maxScore: 12 }));
    expect(c1(done).showcases[0]).toMatchObject({ status: "done", score: 12, maxScore: 12 });
    expect(c1(done).fame.followers).toBeCloseTo(followers * (1 + B.followers));
    const story = done.feed.filter((f) => f.kind === "news" && f.topic === "showcase");
    expect(story[0]?.kind === "news" && story[0].headline).toBe(
      "Acme's Spark clears the whole Pattern Bridge live on stage",
    );
    expect(story[0]?.kind === "news" && story[0].sentiment).toBe(1);
  });

  it("a bad run gets roasted", () => {
    const done = ok(resolve(ok(start(live())), { score: 1, maxScore: 12 }));
    const story = done.feed.filter((f) => f.kind === "news" && f.topic === "showcase");
    expect(story[0]?.kind === "news" && story[0].headline).toBe(
      "Acme's Spark stumbles on the Pattern Bridge live on stage",
    );
    expect(story[0]?.kind === "news" && story[0].sentiment).toBeLessThan(0);
  });

  it("a middling run says how far it got", () => {
    const done = ok(resolve(ok(start(live(), { game: "maze" })), { score: 7, maxScore: 10 }));
    const story = done.feed.filter((f) => f.kind === "news" && f.topic === "showcase");
    expect(story[0]?.kind === "news" && story[0].headline).toBe(
      "Acme's Spark reaches step 7 of the Puzzle Maze live on stage",
    );
  });

  it("a called-off showcase refunds the fee once", () => {
    const s = live();
    const done = ok(resolve(ok(start(s)), { failed: true }));
    expect(c1(done).cash).toBe(c1(s).cash);
    expect(c1(done).showcases[0]?.status).toBe("calledOff");
    expect(resolve(done, { failed: true }).ok).toBe(false);
  });

  it("refuse a result that isn't valid", () => {
    const s = ok(start(live()));
    expect(resolve(s, { score: 13, maxScore: 12 }).ok).toBe(false);
    expect(resolve(s, { score: -1, maxScore: 12 }).ok).toBe(false);
  });

  it("show in the owner's view only", () => {
    const s = ok(start(live()));
    expect(projectView(s, "u1").me.company?.showcases[0]).toMatchObject({
      id: "sh1",
      game: "bridge",
      status: "running",
    });
  });
});
