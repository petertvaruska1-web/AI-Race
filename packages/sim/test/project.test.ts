import { type Command, REGION_IDS, WorldView } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  adoptionShare,
  BALANCE_V2,
  customersOf,
  type Input,
  projectView,
  type StepResult,
  step,
  totalCustomers,
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
    ...BALANCE_V2.defaultPlan,
    mix: { ...BALANCE_V2.defaultPlan.mix },
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
    expect(view.me.company?.cash).toBe(BALANCE_V2.foundingCash - 2 * 41_596);
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
      "customers",
      "customersByRegion",
      "followers",
      "id",
      "logo",
      "name",
      "ownerUserId",
      "ownership",
      "palette",
      "prices",
      "products",
      "publishedVersions",
      "reputation",
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

  it("keeps the AI's measured traits for its owner only", () => {
    let s = ok(
      step(world(), {
        kind: "system",
        event: {
          type: "evalCompleted",
          versionId: "VER_A1",
          scores: { language: 40 },
          overall: 21,
          personalityWords: ["curious"],
          traits: { curious: 0.4, terse: 0.1 },
        },
      }),
    );
    expect(projectView(s, "u1").me.company?.versions[0]?.traits).toEqual({
      curious: 0.4,
      terse: 0.1,
    });
    s = structuredClone(s);
    const v1 = s.companies.CO_A?.versions.VER_A1;
    if (v1) v1.published = true;
    const shared = WorldView.parse(projectView(s, "u2")).companies.find((c) => c.id === "CO_A");
    expect(Object.keys(shared?.publishedVersions[0] ?? {})).not.toContain("traits");
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

const ticks = (s: WorldState, n: number) => {
  let out = s;
  for (let i = 0; i < n; i++) out = ok(step(out, { kind: "tick" }));
  return out;
};
/** CO_A announces "Spark" (VER_A1) on day 13; it goes live on day 16. */
const launch = (s: WorldState) =>
  ok(
    step(
      s,
      cmd(
        "u1",
        {
          type: "launchProduct",
          versionId: "VER_A1",
          name: "Spark",
          plans: ["free", "personal"],
          countdownDays: 3,
        },
        "PROD_A",
      ),
    ),
  );
const feedback = (key: string, scores: [string, number][]): Input =>
  ({
    kind: "system",
    event: {
      type: "feedbackScored",
      productId: "PROD_A",
      key,
      results: scores.map(([group, score]) => ({
        group,
        questionId: `q-${group}`,
        question: "What is 2 + 2?",
        reply: "<b>x</b> <|end|>",
        score,
      })),
    },
  }) as Input;
const acmeFor = (s: WorldState, userId = "u2") => {
  const acme = WorldView.parse(projectView(s, userId)).companies.find((c) => c.id === "CO_A");
  if (!acme) throw new Error("no CO_A");
  return acme;
};

describe("projectView: the company game", () => {
  it("shows the owner its products, plans, customers, satisfaction, ledger and fame", () => {
    let s = ticks(launch(world()), 10);
    s = ok(step(s, feedback("w3", [["students", 0.9]])));
    const company = s.companies.CO_A;
    const me = WorldView.parse(projectView(s, "u1")).me.company;
    expect(me?.products).toEqual([
      {
        id: "PROD_A",
        name: "Spark",
        versionId: "VER_A1",
        status: "live",
        announcedDay: 13,
        launchDay: 16,
        hype: company?.products.PROD_A?.hype,
      },
    ]);
    expect(me?.plans).toEqual({
      free: { productId: "PROD_A" },
      personal: { productId: "PROD_A", price: 10 },
      business: { productId: null, price: 30 },
    });
    const total = totalCustomers(s.market, "CO_A");
    expect(total).toBeGreaterThan(0);
    expect(me?.customers.total).toBe(Math.round(total));
    expect(me?.customers.byGroup.students.personal).toBe(
      Math.round(
        REGION_IDS.reduce(
          (n, r) => n + customersOf(s.market, "CO_A", r, "students", "personal"),
          0,
        ),
      ),
    );
    expect(me?.satisfaction.PROD_A?.students).toBeCloseTo(0.62, 9);
    expect(me?.satisfaction.PROD_A?.everyday).toBe(0.5);
    expect(me?.ledger.current.month).toBe("2027-01");
    expect(me?.ledger.current.costs.training).toBe(2 * 41_596);
    expect(me?.fame).toEqual({
      reputation: Math.round(company?.fame.reputation ?? -1),
      followers: Math.round(company?.fame.followers ?? -1),
    });
  });

  it("shows another company's product once announced, and its AI only from launch day", () => {
    let s = launch(world());
    expect(acmeFor(s).products).toEqual([
      { id: "PROD_A", name: "Spark", status: "countdown", launchDay: 16, version: null },
    ]);
    expect(acmeFor(s).prices).toEqual({ free: false, personal: null, business: null });
    expect(JSON.stringify(projectView(s, "u2"))).not.toContain("VER_A1");
    s = ticks(s, 3);
    expect(acmeFor(s).products[0]).toEqual({
      id: "PROD_A",
      name: "Spark",
      status: "live",
      launchDay: 16,
      version: {
        id: "VER_A1",
        label: "v1",
        createdDay: 13,
        scores: { language: 40 },
        overall: 21,
        personalityWords: ["curious"],
      },
    });
    expect(acmeFor(s).prices).toEqual({ free: true, personal: 10, business: null });
    s = ok(step(s, cmd("u1", { type: "setPlan", plan: "free", productId: null })));
    s = ok(step(s, cmd("u1", { type: "setPlan", plan: "personal", productId: null })));
    s = ok(step(s, cmd("u1", { type: "retireProduct", productId: "PROD_A" })));
    expect(acmeFor(s).products).toEqual([]);
    expect(acmeFor(s).prices).toEqual({ free: false, personal: null, business: null });
  });

  it("rounds other companies' customers to three figures, and fame to whole numbers", () => {
    const s = world();
    s.market.customers.CO_A = { pacific: { everyday: { free: 123_456.7 } } };
    expect(acmeFor(s).customers).toBe(123_000);
    expect(projectView(s, "u1").me.company?.customers.total).toBe(123_457);
    s.market.customers.CO_A = { pacific: { everyday: { free: 56.4 } } };
    expect(acmeFor(s).customers).toBe(56);
    const fame = s.companies.CO_A?.fame;
    if (!fame) throw new Error("no fame");
    fame.reputation = 61.7;
    fame.followers = 1234.4;
    expect(acmeFor(s)).toMatchObject({ reputation: 62, followers: 1234 });
  });

  it("shows everyone the last 50 feed items, replies verbatim, and the spotlights", () => {
    let s = ticks(launch(world()), 3);
    for (let w = 0; w < 30; w++) {
      s = ok(
        step(
          s,
          feedback(`w${w}`, [
            ["students", 0.9],
            ["developers", 0.1],
          ]),
        ),
      );
    }
    expect(s.feed.length).toBeGreaterThan(50);
    for (const user of ["u1", "u2"]) {
      const view = WorldView.parse(projectView(s, user));
      expect(view.feed.map((i) => i.id)).toEqual(s.feed.slice(-50).map((i) => i.id));
      const post = view.feed.find((i) => i.kind === "post");
      expect(post?.kind === "post" && post.reply).toBe("<b>x</b> <|end|>");
      expect(view.spotlights).toEqual(s.spotlights);
    }
  });

  it("shows the market's size", () => {
    const market = projectView(world(), "u1").world.market;
    const share = adoptionShare(BALANCE_V2.economy, 13);
    expect(market.adoption).toBeCloseTo(share, 12);
    const people = REGION_IDS.reduce((n, r) => n + BALANCE_V2.economy.population[r], 0);
    expect(market.adoptable).toBe(Math.round(people * share));
    expect(market.regions.pacific.adoptable).toBe(
      Math.round(BALANCE_V2.economy.population.pacific * share),
    );
  });

  it("shows the owner the last 12 closed months", () => {
    const s = ticks(world(), 400); // 2028-02-17
    expect(s.companies.CO_A?.ledger.months).toHaveLength(13);
    const months = WorldView.parse(projectView(s, "u1")).me.company?.ledger.months ?? [];
    expect(months.map((m) => m.month)).toHaveLength(12);
    expect(months[0]?.month).toBe("2027-02");
    expect(months.at(-1)?.month).toBe("2028-01");
  });
});
