import { type Command, WorldView } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  BALANCE_TEST,
  closeness,
  computeDay,
  createWorld,
  customersByRegion,
  emptyMonth,
  type Input,
  ownedPflops,
  paramCount,
  previewPlan,
  projectView,
  type StepResult,
  servingLoad,
  step,
  totalCustomers,
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
const e = BALANCE_TEST.economy;

function ok(r: StepResult): WorldState {
  if (!r.ok) throw new Error(`rejected: ${r.reason}`);
  return r.state;
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
const c1 = (s: WorldState) => {
  const c = s.companies.c1;
  if (!c) throw new Error("no company");
  return c;
};

/** c1's scored v1 (2 layers of 64) live on every plan. */
function live(): WorldState {
  let s = createWorld({
    worldId: "w1",
    name: "Lab",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 5,
    balance: "balance-test",
    inviteCode: "ABC234",
  });
  const palette = { primary: "#4f8cff", secondary: "#ffb020" };
  s = ok(step(s, cmd({ type: "createCompany", name: "Acme", logo: LOGO, palette }, "c1")));
  s = ok(step(s, cmd({ type: "startWorld" })));
  const plan = { ...BALANCE_TEST.defaultPlan, mix: { ...BALANCE_TEST.defaultPlan.mix } };
  s = ok(step(s, cmd({ type: "startTraining", plan }, "r1")));
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
  s = ok(
    step(
      s,
      cmd(
        {
          type: "launchProduct",
          versionId: "v1",
          name: "Spark",
          plans: ["free", "personal", "business"],
          countdownDays: 3,
        },
        "p1",
      ),
    ),
  );
  return ticks(s, 3);
}

/** Replaces c1's customers with `n` everyday free users in the Pacific. */
function only(s: WorldState, n: number): WorldState {
  const next = structuredClone(s);
  next.market.customers.c1 = { pacific: { everyday: { free: n } } };
  return next;
}

const millions = paramCount(2, 64) / 1e6;
const dollarsFor = (users: number) => users * e.servingCostPerCustomerDayPerMParam * millions;

describe("serving capacity", () => {
  it("serves everyone within the cloud's room, at the price per user Phase 1 charged", () => {
    const s = only(live(), 1000);
    c1(s).ledger.current = emptyMonth(c1(s).ledger.current.month);
    computeDay(s);
    expect(c1(s).compute.served).toBe(1);
    expect(servingLoad(s, "c1")).toBeCloseTo(
      (1000 * millions * e.servingCostPerCustomerDayPerMParam) / BALANCE_TEST.cloudPricePerPflop,
      15,
    );
    expect(c1(s).ledger.current.costs.serving).toBe(Math.round(dollarsFor(1000)));
  });

  it("grows the cloud a fifth a day, so a sudden spike overloads it, once in the news", () => {
    // 50,000 users of the 2 x 64 test model, with the cloud sized to them.
    let s = only(live(), 50_000);
    c1(s).compute.cloudPflops = servingLoad(s, "c1");
    computeDay(s);
    expect(c1(s).compute.served).toBe(1);
    const cloud = c1(s).compute.cloudPflops;
    const satisfied = s.market.satisfaction.p1?.everyday ?? e.feedback.satisfactionStart;
    const stories = () => s.feed.filter((f) => f.kind === "news" && f.topic === "overload");

    s = only(s, 150_000);
    computeDay(s);
    expect(c1(s).compute.served).toBeCloseTo(1.2 / 3, 9);
    expect(c1(s).compute.cloudPflops).toBeCloseTo(cloud * 1.2, 15);
    expect(s.market.satisfaction.p1?.everyday ?? 1).toBeLessThan(satisfied);
    expect(stories()).toHaveLength(1);
    const story = stories()[0];
    expect(story?.kind === "news" && story.headline).toBe(
      "Acme's AI buckles under demand: only 40% of questions answered",
    );

    let days = 1;
    while (c1(s).compute.served < 1 && days < 20) {
      computeDay(s);
      days += 1;
    }
    expect(days).toBe(7); // the spike day and six more: 1.2 ** 7 >= 3 > 1.2 ** 6
    expect(stories()).toHaveLength(1);
  });

  it("makes an overloaded company less appealing than its twin", () => {
    const s = only(live(), 20_000);
    const struggling = structuredClone(s);
    c1(struggling).compute.served = 0.5;
    const a = ticks(s, 1);
    const b = ticks(struggling, 1);
    expect(totalCustomers(b.market, "c1")).toBeLessThan(totalCustomers(a.market, "c1"));
  });

  it("charges nothing and stays at full service with no one to serve", () => {
    const s = only(live(), 0);
    c1(s).ledger.current = emptyMonth(c1(s).ledger.current.month);
    computeDay(s);
    expect(c1(s).compute.served).toBe(1);
    expect(Number.isFinite(c1(s).compute.cloudPflops)).toBe(true);
    expect(c1(s).ledger.current.costs.serving).toBe(0);
    const closed = only(live(), 50_000);
    const retired = ok(step(closed, cmd({ type: "setPlan", plan: "free", productId: null })));
    retired.market.customers.c1 = { pacific: { everyday: { free: 50_000 } } };
    computeDay(retired);
    expect(servingLoad(retired, "c1")).toBe(0);
    expect(c1(retired).compute.served).toBe(1);
  });

  it("shows its numbers to its owner only", () => {
    const s = ticks(only(live(), 1000), 1);
    const view = WorldView.parse(projectView(s, "u1"));
    expect(view.me.company?.compute).toEqual({
      cloudPflops: c1(s).compute.cloudPflops,
      ownedPflops: 0,
      trainingShare: 0,
      loadPflops: c1(s).compute.loadPflops,
      served: 1,
    });
  });
});

const BIG = 50_000_000;

/** `s` with an open Pacific data center, as if built 60 days ago. */
function withCenter(s: WorldState): WorldState {
  const next = structuredClone(s);
  c1(next).dataCenters.dc1 = {
    id: "dc1",
    region: "pacific",
    startDay: Math.max(0, next.day - 60),
    openDay: next.day,
    pflopsPerDay: e.dataCenter.pflopsPerDay,
  };
  return next;
}

describe("data centers", () => {
  it("refuse to be built without the money, or past the most a company may own", () => {
    const s = live();
    const r = step(s, cmd({ type: "buildDataCenter", region: "pacific" }, "dc1"));
    expect(r.ok ? "" : r.reason).toMatch(/^A data center costs \$1,500,000\. You have \$[\d,]+\.$/);
    let rich = structuredClone(s);
    c1(rich).cash = BIG;
    for (let i = 1; i <= e.dataCenter.maxPerCompany; i++) {
      rich = ok(step(rich, cmd({ type: "buildDataCenter", region: "pacific" }, `dc${i}`)));
    }
    const more = step(rich, cmd({ type: "buildDataCenter", region: "pacific" }, "dc9"));
    expect(more.ok ? "" : more.reason).toBe("You can own at most 6 data centers.");
  });

  it("are paid for at once, open after their build time with the news, then cost upkeep daily", () => {
    let s = live();
    c1(s).cash = BIG;
    const built = s.day;
    s = ok(step(s, cmd({ type: "buildDataCenter", region: "pacific" }, "dc1")));
    expect(c1(s).cash).toBe(BIG - e.dataCenter.cost);
    expect(c1(s).ledger.current.costs.dataCenterBuild).toBe(e.dataCenter.cost);
    s = ticks(s, e.dataCenter.buildDays - 1);
    expect(ownedPflops(c1(s), s.day)).toBe(0);
    s = ticks(s, 1);
    expect(s.day).toBe(built + e.dataCenter.buildDays);
    expect(ownedPflops(c1(s), s.day)).toBe(e.dataCenter.pflopsPerDay);
    const opened = s.feed.filter((f) => f.kind === "news" && f.topic === "dataCenter");
    expect(opened.map((f) => (f.kind === "news" ? f.headline : ""))).toEqual([
      "Acme opens a data center in the Pacific",
    ]);
    const months = () => [...c1(s).ledger.months, c1(s).ledger.current];
    const upkeep = () => months().reduce((n, m) => n + m.costs.dataCenterUpkeep, 0);
    const before = upkeep();
    s = ticks(s, 30);
    expect(upkeep() - before).toBe(e.dataCenter.upkeepPerMonth);
    expect(s.feed.filter((f) => f.kind === "news" && f.topic === "dataCenter")).toHaveLength(1);
  });

  it("serve their customers for less than the cloud", () => {
    const cloudOnly = only(live(), 100_000);
    c1(cloudOnly).compute.cloudPflops = Number.POSITIVE_INFINITY;
    const owned = withCenter(cloudOnly);
    for (const s of [cloudOnly, owned]) {
      c1(s).ledger.current = emptyMonth(c1(s).ledger.current.month);
      computeDay(s);
    }
    expect(c1(owned).compute.served).toBe(1);
    expect(c1(owned).ledger.current.costs.serving).toBe(
      Math.round(dollarsFor(100_000) * e.compute.ownedServingPriceShare),
    );
    expect(c1(owned).ledger.current.costs.serving).toBeLessThan(
      c1(cloudOnly).ledger.current.costs.serving,
    );
  });

  it("answer faster nearby, winning customers in their region and its neighbours", () => {
    const s = live();
    const near = ticks(withCenter(s), 10);
    const far = ticks(s, 10);
    const inRegion = (w: WorldState, r: "pacific" | "mountain") =>
      customersByRegion(w.market, "c1")[r] ?? 0;
    expect(closeness(c1(near), "pacific", near.day, e)).toBe(1);
    expect(closeness(c1(near), "mountain", near.day, e)).toBe(e.nearby.neighbor);
    expect(closeness(c1(near), "newEngland", near.day, e)).toBe(0);
    expect(inRegion(near, "pacific")).toBeGreaterThan(inRegion(far, "pacific"));
    expect(inRegion(near, "mountain")).toBeGreaterThan(inRegion(far, "mountain"));
  });

  it("lend a share to training: the next run is shorter and cheaper, serving room smaller", () => {
    const plan = { ...BALANCE_TEST.defaultPlan, mix: { ...BALANCE_TEST.defaultPlan.mix } };
    const owned = e.dataCenter.pflopsPerDay * 0.5;
    const plain = previewPlan(BALANCE_TEST, null, plan);
    const lent = previewPlan(BALANCE_TEST, null, plan, owned);
    if (!plain.ok || !lent.ok) throw new Error("no preview");
    const pflop = plain.preview.flops / 1e15;
    const cloud = BALANCE_TEST.cloudPflopsPerDay;
    expect(lent.preview.durationDays).toBe(Math.max(1, Math.ceil(pflop / (cloud + owned))));
    expect(lent.preview.cost).toBe(
      Math.round((pflop * BALANCE_TEST.cloudPricePerPflop * cloud) / (cloud + owned)),
    );
    expect(lent.preview.cost).toBeLessThan(plain.preview.cost);

    // 4 million users need more than half a center: lent away, the rest overloads the cloud.
    const busy = withCenter(only(live(), 4_000_000));
    computeDay(busy);
    expect(c1(busy).compute.served).toBe(1);
    const sharing = ok(
      step(withCenter(only(live(), 4_000_000)), cmd({ type: "setTrainingShare", share: 0.5 })),
    );
    expect(c1(sharing).compute.trainingShare).toBe(0.5);
    computeDay(sharing);
    expect(c1(sharing).compute.served).toBeLessThan(1);
  });

  it("refuse a share for training outside 0 to 1", () => {
    const s = live();
    for (const share of [-0.1, 1.5]) {
      const r = step(s, cmd({ type: "setTrainingShare", share }));
      expect(r.ok ? "" : r.reason).toBe("Choose a share from 0% to 100%.");
    }
    // Not a number at all never passes the command check.
    expect(step(s, cmd({ type: "setTrainingShare", share: Number.NaN })).ok).toBe(false);
    const r = ok(step(s, cmd({ type: "setTrainingShare", share: 0.333 })));
    expect(c1(r).compute.trainingShare).toBe(0.33);
  });

  it("show to their owner only", () => {
    const s = withCenter(live());
    const view = WorldView.parse(projectView(s, "u1"));
    expect(view.me.company?.dataCenters).toEqual([
      {
        id: "dc1",
        region: "pacific",
        startDay: Math.max(0, s.day - 60),
        openDay: s.day,
        open: true,
      },
    ]);
    expect(view.me.company?.compute.ownedPflops).toBe(e.dataCenter.pflopsPerDay);
  });
});
