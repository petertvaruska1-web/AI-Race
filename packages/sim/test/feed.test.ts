import type { Command, GroupId, PlanId } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  BALANCE_TEST,
  type CompanyState,
  createWorld,
  type FeedbackResult,
  type Input,
  type NewsItem,
  newsDay,
  type PostItem,
  type StepResult,
  step,
  type WorldState,
} from "../src/index.ts";

const LOGO = { shape: "circle", symbol: "spark", initials: "AC" } as const;
const SCORES = {
  language: 60,
  reasoning: 8,
  pattern: 80,
  knowledge: 44,
  coding: 38,
  creativity: 2,
  consistency: 37,
  instruction: 63,
};

function ok(r: StepResult): WorldState {
  if (!r.ok) throw new Error(`rejected: ${r.reason}`);
  return r.state;
}
function rejected(r: StepResult): string {
  if (r.ok) throw new Error("expected a rejection");
  return r.reason;
}
const cmd = (c: unknown, newId?: string): Input => ({
  kind: "command",
  userId: "u1",
  cmd: c as Command,
  ...(newId ? { newId } : {}),
});
const sys = (event: unknown): Input => ({ kind: "system", event }) as Input;
const ticks = (s: WorldState, n: number) => {
  let out = s;
  for (let i = 0; i < n; i++) out = ok(step(out, { kind: "tick" }));
  return out;
};

/** A running world on day 0 where u1 owns company c1 ("Acme"). */
function world(): WorldState {
  let s = createWorld({
    worldId: "w1",
    name: "Lab",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 5,
    balance: "balance-test",
    inviteCode: "ABC234",
  });
  s = ok(
    step(
      s,
      cmd(
        {
          type: "createCompany",
          name: "Acme",
          logo: LOGO,
          palette: { primary: "#4f8cff", secondary: "#ffb020" },
        },
        "c1",
      ),
    ),
  );
  return ok(step(s, cmd({ type: "startWorld" })));
}

/** c1 has a scored v1 and announced "Spark" on free and personal, live in 3 days. */
function countdown(): WorldState {
  const plan = { ...BALANCE_TEST.defaultPlan, mix: { ...BALANCE_TEST.defaultPlan.mix } };
  let s = ok(step(world(), cmd({ type: "startTraining", plan }, "r1")));
  s = ok(
    step(
      s,
      sys({
        type: "trainingCompleted",
        runId: "r1",
        versionId: "v1",
        result: { steps: 8, finalLoss: 4, heldoutLosses: {}, unstable: false },
      }),
    ),
  );
  s = ticks(s, (s.companies.c1?.runs.r1?.endDay ?? 0) - s.day);
  s = ok(
    step(
      s,
      sys({
        type: "evalCompleted",
        versionId: "v1",
        scores: SCORES,
        overall: 42,
        personalityWords: [],
      }),
    ),
  );
  return ok(
    step(
      s,
      cmd(
        {
          type: "launchProduct",
          versionId: "v1",
          name: "Spark",
          plans: ["free", "personal"],
          countdownDays: 3,
        },
        "p1",
      ),
    ),
  );
}

const live = () => ticks(countdown(), 3);

const company = (s: WorldState): CompanyState => {
  const c = s.companies.c1;
  if (!c) throw new Error("no company");
  return c;
};

const result = (
  group: GroupId,
  score: number,
  reply = `A reply for ${group}.`,
): FeedbackResult => ({
  group,
  questionId: `q-${group}`,
  question: `What would ${group} ask?`,
  reply,
  score,
});
const scored = (key: string, results: FeedbackResult[], productId = "p1"): Input =>
  sys({ type: "feedbackScored", productId, key, results });
const BATCH = [result("students", 0.9), result("everyday", 0.5), result("developers", 0.1)];

const posts = (s: WorldState) => s.feed.filter((i): i is PostItem => i.kind === "post");
const news = (s: WorldState, topic: NewsItem["topic"]) =>
  s.feed.filter((i): i is NewsItem => i.kind === "news" && i.topic === topic);
const satisfaction = (s: WorldState, group: GroupId) =>
  s.market.satisfaction.p1?.[group] ?? BALANCE_TEST.economy.feedback.satisfactionStart;

/** Puts `n` customers of `group` on `plan`, replacing what the market had. */
function give(s: WorldState, group: GroupId, plan: PlanId, n: number): void {
  s.market.customers.c1 = { pacific: { [group]: { [plan]: n } } };
}

describe("customer feedback", () => {
  it("turns the best and the worst reply of a batch into exactly two posts", () => {
    const s = ok(step(live(), scored("w5", BATCH)));
    const p = posts(s);
    expect(p).toHaveLength(2);
    expect(p.map((x) => [x.author.group, x.tone])).toEqual([
      ["students", "praise"],
      ["developers", "roast"],
    ]);
    expect(p.every((x) => x.companyId === "c1" && x.productName === "Spark")).toBe(true);
    expect(news(s, "viral")).toHaveLength(0);
  });

  it("quotes the reply and the question exactly as received", () => {
    const reply = "<b>x</b> <|end|>";
    const s = ok(
      step(live(), scored("w5", [result("students", 0.9, reply), result("developers", 0.1)])),
    );
    const praise = posts(s).find((x) => x.tone === "praise");
    expect(praise?.reply).toBe(reply);
    expect(praise?.question).toBe("What would students ask?");
  });

  it("raises the praised group's satisfaction and lowers the roasted group's", () => {
    const before = live();
    const after = ok(step(before, scored("w5", BATCH)));
    const alpha = BALANCE_TEST.economy.feedback.satisfactionAlpha;
    const start = satisfaction(before, "students");
    expect(satisfaction(after, "students")).toBeCloseTo(start + alpha * (0.9 - start), 9);
    expect(satisfaction(after, "developers")).toBeLessThan(satisfaction(before, "developers"));
    expect(satisfaction(after, "everyday")).toBeCloseTo(satisfaction(before, "everyday"), 9);
  });

  it("applies a batch once: the same product and key again is refused", () => {
    const once = ok(step(live(), scored("w5", BATCH)));
    expect(rejected(step(once, scored("w5", BATCH)))).toMatch(/already/i);
    expect(ok(step(once, scored("w6", BATCH))).feed.length).toBeGreaterThan(once.feed.length);
  });

  it("refuses feedback for a product that is not on sale, and odd batches", () => {
    expect(rejected(step(countdown(), scored("launch", BATCH)))).toMatch(/sale/i);
    expect(rejected(step(live(), scored("w5", BATCH, "nope")))).toMatch(/unknown/i);
    expect(rejected(step(live(), scored("w5", [result("students", 1.5)])))).toMatch(/valid/i);
    expect(rejected(step(live(), scored("w5", [result("aliens" as GroupId, 0.5)])))).toMatch(
      /valid/i,
    );
  });

  it("picks each post's tone from its score", () => {
    let s = live();
    [0.75, 0.25, 0.74, 0.26].forEach((score, i) => {
      s = ok(step(s, scored(`w${i}`, [result("students", score)])));
    });
    expect(posts(s).map((x) => x.tone)).toEqual(["praise", "roast", "joke", "joke"]);
    posts(s).forEach((x, i) => {
      expect(x.sentiment).toBeCloseTo([0.5, -0.5, 0.48, -0.48][i] ?? 0, 9);
    });
  });

  it("gives about half the posts to the group's spotlight customer", () => {
    let s = live();
    for (let w = 0; w < 40; w++) s = ok(step(s, scored(`w${w}`, BATCH)));
    const p = posts(s);
    const spot = p.filter((x) => x.author.spotlightId !== null);
    expect(spot.length).toBeGreaterThan(p.length * 0.25);
    expect(spot.length).toBeLessThan(p.length * 0.75);
    for (const x of spot) {
      const person = s.spotlights.find((sp) => sp.id === x.author.spotlightId);
      expect(person).toMatchObject({ name: x.author.name, handle: x.author.handle });
      expect(person?.group).toBe(x.author.group);
    }
    for (const x of p) expect(x.author.handle).toMatch(/^@[a-z0-9_]+$/);
  });

  it("never lets the feed grow past its cap, dropping the oldest", () => {
    let s = live();
    for (let w = 1; w <= 101; w++) s = ok(step(s, scored(`w${w}`, BATCH)));
    expect(s.feed).toHaveLength(BALANCE_TEST.economy.feedCap);
    expect(posts(s).at(-1)?.tone).toBe("roast");
    expect(new Set(s.feed.map((i) => i.id)).size).toBe(s.feed.length);
  });
});

describe("news", () => {
  it("announces a launch once, on its day, with its real score", () => {
    let s = ticks(countdown(), 2);
    expect(news(s, "launch")).toHaveLength(0);
    s = ticks(s, 1);
    const [story] = news(s, "launch");
    expect(story?.headline).toBe("Acme launches Spark, scoring 80 in patterns");
    expect(story?.companyId).toBe("c1");
    expect(story?.sentiment).toBeGreaterThan(0);
    expect(news(ticks(s, 5), "launch")).toHaveLength(1);
  });

  it("announces 1,000 customers once, however often it is crossed", () => {
    const s = world();
    give(s, "students", "free", 1500);
    newsDay(s);
    expect(news(s, "customers").map((n) => n.headline)).toEqual(["Acme passes 1,000 customers"]);
    newsDay(s);
    give(s, "students", "free", 900);
    newsDay(s);
    give(s, "students", "free", 1500);
    newsDay(s);
    expect(news(s, "customers")).toHaveLength(1);
    give(s, "students", "free", 250_000);
    newsDay(s);
    expect(news(s, "customers").at(-1)?.headline).toBe("Acme passes 100,000 customers");
    expect(news(s, "customers")).toHaveLength(2);
  });

  it("announces the first paying customer once", () => {
    const s = world();
    give(s, "everyday", "personal", 0.6);
    newsDay(s);
    expect(news(s, "firstPaying")).toHaveLength(0);
    give(s, "everyday", "personal", 1.2);
    newsDay(s);
    newsDay(s);
    expect(news(s, "firstPaying").map((n) => n.headline)).toEqual([
      "Acme gets its first paying customer",
    ]);
  });

  it("makes one viral story of the most extreme post when followers are high", () => {
    const s = live();
    company(s).fame.followers = 5000;
    const next = ok(step(s, scored("w5", [result("students", 0.95), result("developers", 0.02)])));
    const viral = news(next, "viral");
    expect(viral).toHaveLength(1);
    const roast = posts(next).find((x) => x.tone === "roast");
    expect(viral[0]?.headline).toBe(`${roast?.author.handle}'s roast of Spark is going viral`);
    expect(viral[0]?.sentiment).toBeLessThan(0);
    const { viralFollowers } = BALANCE_TEST.economy.fame;
    expect(company(next).fame.followers).toBe(5000 + viralFollowers / 10);
  });
});

describe("fame", () => {
  it("adds followers for the launch's hype", () => {
    let s = ticks(countdown(), 2);
    const hype = company(s).products.p1?.hype ?? 0;
    expect(hype).toBeGreaterThan(0);
    const before = company(s).fame.followers;
    s = ticks(s, 1);
    const { followersPerHype, followersDecayPerDay } = BALANCE_TEST.economy.fame;
    expect(company(s).fame.followers).toBeCloseTo(
      (before + followersPerHype * hype) * (1 - followersDecayPerDay),
      9,
    );
  });

  it("keeps reputation within 0..100 as customers turn unhappy and then happy", () => {
    const all = (score: number) =>
      (
        ["students", "creators", "developers", "businesses", "everyday", "researchers"] as const
      ).map((g) => result(g, score));
    let s = live();
    const start = company(s).fame.reputation;
    for (let w = 0; w < 30; w++) {
      s = ticks(ok(step(s, scored(`low${w}`, all(0)))), 7);
      expect(company(s).fame.reputation).toBeGreaterThanOrEqual(0);
    }
    const low = company(s).fame.reputation;
    expect(low).toBeLessThan(start);
    for (let w = 0; w < 30; w++) {
      s = ticks(ok(step(s, scored(`high${w}`, all(1)))), 7);
      expect(company(s).fame.reputation).toBeLessThanOrEqual(100);
    }
    expect(company(s).fame.reputation).toBeGreaterThan(low);
  });
});
