import { describe, expect, it } from "vitest";
import { type CompanyState, getBalance, projectView, step, type WorldState } from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys, ticks } from "./pack.ts";

const c1 = (s: WorldState) => s.companies.c1 as CompanyState;
const A = getBalance("balance-test").economy.apps;

/** c1 is rich and sells "Spark" (v1, knowledge `knowledge`), live. */
function live(knowledge: number | null = 64): WorldState {
  const s = structuredClone(runningWorld("balance-test"));
  c1(s).cash = 5_000_000;
  c1(s).startCash = 5_000_000;
  c1(s).versions.v1 = {
    id: "v1",
    label: "v1",
    runId: "r1",
    parentId: null,
    nLayer: 2,
    dModel: 64,
    ctxLen: 64,
    createdDay: s.day,
    trainedTokens: 1,
    scores: { knowledge },
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
const fund = (s: WorldState, kind = "homework", id = "a1") =>
  step(s, cmd({ type: "fundApp", kind, productId: "p1" }, id));
const review = (s: WorldState, key: string, scores: number[], appId = "a1") =>
  step(
    s,
    sys({
      type: "appReviewed",
      appId,
      key,
      reviews: scores.map((score, k) => ({ prompt: `Q${k}?`, reply: `A${k}`, score })),
    }),
  );

describe("partner apps", () => {
  it("charge their setup and start with no users and three stars", () => {
    const s = live();
    const after = ok(fund(s));
    expect(c1(after).cash).toBe(c1(s).cash - A.kinds.homework.setup);
    expect(c1(after).ledger.current.costs.appCosts).toBe(A.kinds.homework.setup);
    expect(c1(after).apps.a1).toMatchObject({
      kind: "homework",
      productId: "p1",
      users: 0,
      rating: 3,
      status: "live",
    });
  });

  it("need a live product of yours, one app of a kind, and the money", () => {
    const s = live();
    expect(
      rejected(step(s, cmd({ type: "fundApp", kind: "homework", productId: "nope" }, "a1"))),
    ).toMatch(/on sale/);
    const once = ok(fund(s));
    expect(rejected(fund(once, "homework", "a2"))).toMatch(/already/);
    const poor = structuredClone(s);
    c1(poor).cash = 1;
    expect(rejected(fund(poor))).toMatch(/costs/);
  });

  it("grow by the AI's measured skill: a better score, more users", () => {
    const good = ticks(ok(fund(live(80))), 30);
    const weak = ticks(ok(fund(live(20))), 30);
    const none = ticks(ok(fund(live(null))), 30);
    expect(c1(good).apps.a1?.users).toBeGreaterThan(c1(weak).apps.a1?.users ?? 0);
    expect(c1(weak).apps.a1?.users).toBeGreaterThan(0);
    expect(c1(none).apps.a1?.users).toBeGreaterThan(0); // unmeasured: a little
  });

  it("book a revenue share and upkeep each day", () => {
    const s = ticks(ok(fund(live(80))), 20);
    const m = c1(s).ledger.current;
    expect(m.income.appRevenue).toBeGreaterThan(0);
    expect(m.costs.appCosts).toBeGreaterThan(A.kinds.homework.setup);
  });

  it("post real-reply reviews with stars, and the rating follows them", () => {
    const s = ok(fund(live()));
    const r = ok(review(s, "w1", [1, 0.9]));
    const app = c1(r).apps.a1;
    expect(app?.reviews.map((x) => x.stars)).toEqual([5, 5]);
    expect(app?.rating).toBe(5);
    const post = r.feed.filter((f) => f.kind === "post").at(-1);
    expect(post?.kind === "post" && post.stars).toBe(5);
    expect(post?.kind === "post" && post.productName).toBe("Homework Helper");
    expect(post?.kind === "post" && ["A0", "A1"].includes(post.reply)).toBe(true);
    expect(review(r, "w1", [0]).ok).toBe(false); // a batch counts once
    const bad = ok(review(r, "w2", [0, 0.1]));
    expect(c1(bad).apps.a1?.rating).toBeLessThan(5);
  });

  it("stop costing when closed, and pause when their product leaves sale", () => {
    const s = ticks(ok(fund(live(80))), 10);
    const closed = ok(step(s, cmd({ type: "closeApp", appId: "a1" })));
    expect(c1(closed).apps.a1?.status).toBe("closed");
    const later = ticks(closed, 5);
    expect(c1(later).ledger.current.costs.appCosts).toBe(c1(closed).ledger.current.costs.appCosts);
    const retired = structuredClone(s);
    const p1 = c1(retired).products.p1;
    if (p1) p1.status = "retired"; // its AI left sale
    const after = ticks(retired, 10);
    expect(c1(after).apps.a1?.users).toBeLessThan(c1(s).apps.a1?.users ?? 0);
  });

  it("show to their owner", () => {
    const s = ok(review(ok(fund(live())), "w1", [0.5]));
    const app = projectView(s, "u1").me.company?.apps[0];
    expect(app).toMatchObject({ id: "a1", kind: "homework", productName: "Spark", status: "live" });
    expect(app?.reviews[0]).toMatchObject({ reply: "A0", stars: 3 });
    expect(projectView(s, "u1").me.company?.shop.apps.find((x) => x.kind === "code")?.skill).toBe(
      "coding",
    );
  });
});
