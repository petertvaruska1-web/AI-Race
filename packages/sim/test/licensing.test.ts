import type { GroupId } from "@airace/protocol";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  activeLicense,
  type CompanyState,
  type Input,
  projectView,
  step,
  upgradeState,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys, TICK, ticks, withRivals } from "./pack.ts";

const c = (s: WorldState, id: string) => s.companies[id] as CompanyState;

/** `s` with `userId` joined and founding company `companyId`. */
function withPlayer(s0: WorldState, userId: string, companyId: string, name: string): WorldState {
  let s = ok(step(s0, sys({ type: "memberJoined", userId, displayName: name })));
  s = ok(
    step(
      s,
      cmd(
        {
          type: "createCompany",
          name,
          logo: { shape: "circle", symbol: "spark", initials: name.slice(0, 2).toUpperCase() },
          palette: { primary: "#3a86ff", secondary: "#ffbe0b" },
        },
        companyId,
        userId,
      ),
    ),
  );
  return s;
}

/** Puts a scored AI `versionId` on sale as product `productId`, powering the free and personal plans. */
export function selling(
  s: WorldState,
  companyId: string,
  versionId: string,
  productId: string,
  name: string,
  knowledge = 50,
): void {
  const company = c(s, companyId);
  company.versions[versionId] = {
    id: versionId,
    label: "v1",
    runId: "r",
    parentId: null,
    nLayer: 4,
    dModel: 128,
    ctxLen: 256,
    createdDay: s.day,
    trainedTokens: 1_000_000,
    scores: {
      language: 50,
      reasoning: 50,
      pattern: 50,
      knowledge,
      coding: 50,
      creativity: 50,
      consistency: 50,
      instruction: 50,
    },
    overall: 50,
    personalityWords: [],
    traits: { verbosity: 0.5 },
    published: false,
    packKey: null,
  };
  company.products[productId] = {
    id: productId,
    name,
    versionId,
    status: "live",
    announcedDay: s.day,
    launchDay: s.day,
    hype: 0,
    launchPlans: ["free", "personal"],
    feedbackKeys: [],
    keynote: null,
  };
  company.plans.free.productId = productId;
  company.plans.personal.productId = productId;
}

/** Ana (c1) sells Pengu (v1); Ben (c2) sells Owl (v2); Cy (c3) sells Lark (v3). */
export function market3(): WorldState {
  let s = withPlayer(runningWorld(), "u2", "c2", "Beta");
  s = withPlayer(s, "u3", "c3", "Gamma");
  selling(s, "c1", "v1", "p1", "Pengu");
  selling(s, "c2", "v2", "p2", "Owl");
  selling(s, "c3", "v3", "p3", "Lark");
  return s;
}

type Terms = {
  versionId?: string;
  priceCents?: number;
  months?: number;
  groups?: GroupId[];
  exclusive?: boolean;
};
export const offer = (s: WorldState, t: Terms = {}, userId = "u1") =>
  step(
    s,
    cmd(
      {
        type: "offerLicense",
        versionId: t.versionId ?? "v1",
        priceCents: t.priceCents ?? 300,
        months: t.months ?? 6,
        groups: t.groups ?? ["students"],
        exclusive: t.exclusive ?? false,
      },
      undefined,
      userId,
    ),
  );
export const take = (
  s: WorldState,
  offerId: string,
  productId: string,
  userId: string,
  groups: GroupId[] = ["students"],
) => step(s, cmd({ type: "takeLicense", offerId, productId, groups }, undefined, userId));

describe("license offers", () => {
  it("are public: every player sees the terms", () => {
    const s = ok(
      offer(market3(), { priceCents: 250, months: 12, groups: ["students", "everyday"] }),
    );
    const seen = projectView(s, "u2").licenseOffers;
    expect(seen).toEqual([
      {
        id: s.licenses.offers[0]?.id,
        licensorId: "c1",
        versionId: "v1",
        aiName: "Pengu",
        priceCents: 250,
        months: 12,
        groups: ["students", "everyday"],
        exclusive: false,
        day: s.day,
      },
    ]);
    expect(projectView(s, "u1").licenseOffers).toEqual(seen);
  });

  it("refuses prices outside $0.10–$20, an AI not on sale, and a second offer of one AI", () => {
    const s = market3();
    const price = "Prices are $0.10–$20 a customer a month.";
    expect(rejected(offer(s, { priceCents: 9 }))).toBe(price);
    expect(rejected(offer(s, { priceCents: 2001 }))).toBe(price);
    expect(rejected(offer(s, { versionId: "v2" }))).toBe("Pick one of your AIs on sale.");
    expect(rejected(offer(s, { months: 4 }))).toBe("That request isn't valid.");
    const once = ok(offer(s));
    expect(rejected(offer(once))).toBe("You already offer that AI. Withdraw that offer first.");
  });

  it("can be withdrawn by their maker only", () => {
    const s = ok(offer(market3()));
    const offerId = s.licenses.offers[0]?.id as string;
    expect(rejected(step(s, cmd({ type: "withdrawOffer", offerId }, undefined, "u2")))).toBe(
      "That offer isn't yours.",
    );
    const gone = ok(step(s, cmd({ type: "withdrawOffer", offerId })));
    expect(gone.licenses.offers).toEqual([]);
  });
});

describe("taking a license", () => {
  it("makes a deal for the chosen product and groups", () => {
    let s = ok(offer(market3(), { groups: ["students", "everyday"], months: 3 }));
    const offerId = s.licenses.offers[0]?.id as string;
    s = ok(take(s, offerId, "p2", "u2", ["students"]));
    expect(s.licenses.deals).toEqual([
      {
        id: s.licenses.deals[0]?.id,
        licensorId: "c1",
        licenseeId: "c2",
        versionId: "v1",
        aiName: "Pengu",
        productId: "p2",
        priceCents: 300,
        groups: ["students"],
        startDay: s.day,
        endDay: s.day + 90,
        exclusive: false,
        owed: 0,
      },
    ]);
    expect(activeLicense(s, "p2", "students")?.licensorId).toBe("c1");
    expect(activeLicense(s, "p2", "everyday")).toBeNull();
    expect(activeLicense(s, "p1", "students")).toBeNull();
    // A shared offer stays open for others.
    expect(s.licenses.offers).toHaveLength(1);
    expect(ok(take(s, offerId, "p3", "u3")).licenses.deals).toHaveLength(2);
  });

  it("closes an exclusive offer: the second take is refused", () => {
    let s = ok(offer(market3(), { exclusive: true }));
    const offerId = s.licenses.offers[0]?.id as string;
    s = ok(take(s, offerId, "p2", "u2"));
    expect(s.licenses.offers).toEqual([]);
    expect(s.licenses.deals[0]?.exclusive).toBe(true);
    expect(rejected(take(s, offerId, "p3", "u3"))).toBe("That offer has been taken.");
    // While it runs, nobody else can license that AI.
    expect(rejected(offer(s))).toBe("That AI is licensed exclusively.");
  });

  it("refuses your own AI, someone else's product, groups the offer lacks and a group already licensed", () => {
    const s = ok(offer(market3(), { groups: ["students", "everyday"] }));
    const offerId = s.licenses.offers[0]?.id as string;
    expect(rejected(take(s, offerId, "p1", "u1"))).toBe("You can't license your own AI.");
    expect(rejected(take(s, offerId, "p3", "u2"))).toBe("Pick one of your AIs on sale.");
    expect(rejected(take(s, offerId, "p2", "u2", ["coders" as GroupId]))).toBe(
      "That request isn't valid.",
    );
    expect(rejected(take(s, offerId, "p2", "u2", ["developers"]))).toBe(
      "That offer doesn't cover those groups.",
    );
    const once = ok(take(s, offerId, "p2", "u2"));
    expect(rejected(take(once, offerId, "p2", "u2"))).toBe(
      "That AI already uses a licensed AI for one of those groups.",
    );
  });
});

describe("proposals", () => {
  const propose = (s: WorldState, versionId = "v1", userId = "u2", productId = "p2") =>
    step(
      s,
      cmd(
        {
          type: "proposeLicense",
          versionId,
          productId,
          priceCents: 500,
          months: 6,
          groups: ["researchers"],
        },
        undefined,
        userId,
      ),
    );
  const answer = (s: WorldState, accept: boolean, userId = "u1") =>
    step(
      s,
      cmd(
        { type: "answerProposal", proposalId: s.licenses.proposals[0]?.id, accept },
        undefined,
        userId,
      ),
    );

  it("make a deal when the maker accepts", () => {
    let s = ok(propose(market3()));
    expect(s.licenses.proposals).toHaveLength(1);
    // Only the two companies see it.
    expect(projectView(s, "u1").me.company?.licenses.proposals).toHaveLength(1);
    expect(projectView(s, "u2").me.company?.licenses.proposals).toHaveLength(1);
    expect(projectView(s, "u3").me.company?.licenses.proposals).toEqual([]);
    expect(rejected(answer(s, true, "u2"))).toBe("That request isn't yours to answer.");
    s = ok(answer(s, true));
    expect(s.licenses.proposals).toEqual([]);
    expect(s.licenses.deals).toHaveLength(1);
    expect(s.licenses.deals[0]).toMatchObject({
      licensorId: "c1",
      licenseeId: "c2",
      productId: "p2",
      priceCents: 500,
      groups: ["researchers"],
      endDay: s.day + 180,
    });
    expect(projectView(s, "u2").me.company?.licenses.deals[0]).toMatchObject({
      role: "licensee",
      otherName: "Acme",
      aiName: "Pengu",
      productName: "Owl",
    });
  });

  it("make no deal when declined", () => {
    const s = ok(answer(ok(propose(market3())), false));
    expect(s.licenses.proposals).toEqual([]);
    expect(s.licenses.deals).toEqual([]);
  });

  it("refuse your own AI", () => {
    expect(rejected(propose(market3(), "v1", "u1", "p1"))).toBe("You can't license your own AI.");
  });
});

describe("upgraded worlds", () => {
  it("have no offers, proposals or deals", () => {
    const old = structuredClone(market3()) as unknown as Record<string, unknown>;
    delete old.licenses;
    delete old.sim10FromSeq;
    for (const company of Object.values(old.companies as Record<string, Record<string, unknown>>)) {
      const ledger = company.ledger as { current: { income: Record<string, number> } };
      delete ledger.current.income.licenseIncome;
    }
    const s = upgradeState(old, "sim-9");
    expect(s.licenses).toEqual({ offers: [], proposals: [], deals: [], seq: 0, lastRequest: {} });
    expect(s.sim10FromSeq).toBe(s.seq);
    expect(c(s, "c1").ledger.current.income.licenseIncome).toBe(0);
    expect(c(s, "c1").ledger.current.costs.licenses).toBe(0);
  });
});

/** Ben (c2) takes Ana's Pengu (v1) for Owl's students at $3 a customer a month. */
function licensed(s0 = market3(), groups: GroupId[] = ["students"]): WorldState {
  let s = ok(offer(s0, { groups, priceCents: 300, months: 3 }));
  s = ok(take(s, s.licenses.offers[0]?.id as string, "p2", "u2", groups));
  return s;
}

/** Sets c2's students on the free plan in one region to `n` (and nobody elsewhere). */
function students(s: WorldState, n: number): void {
  s.market.customers.c2 = { pacific: { students: { free: n } } };
}

const studentsOf = (s: WorldState, id: string) => {
  let n = 0;
  for (const byGroup of Object.values(s.market.customers[id] ?? {})) {
    for (const n2 of Object.values(byGroup?.students ?? {})) n += n2 ?? 0;
  }
  return n;
};

describe("licensed AIs in the market", () => {
  it("win a licensee's students with the licensed AI's knowledge", () => {
    const base = market3();
    // Owl knows little; Pengu knows a lot.
    selling(base, "c2", "v2", "p2", "Owl", 5);
    selling(base, "c1", "v1", "p1", "Pengu", 95);
    for (const s of [base]) {
      for (const p of ["p1", "p2", "p3"]) {
        const company = Object.values(s.companies).find((x) => x.products[p]);
        const product = company?.products[p];
        if (product) product.launchDay = s.day + 1; // tomorrow's launch spreads the word
      }
    }
    const without = ticks(base, 40);
    const withLicense = ticks(licensed(structuredClone(base)), 40);
    expect(studentsOf(withLicense, "c2")).toBeGreaterThan(1.2 * studentsOf(without, "c2"));
  });
});

describe("royalties", () => {
  it("build up a day at a time and are paid at the month close, to both sides", () => {
    let s = licensed();
    const startMonth = c(s, "c2").ledger.current.month;
    let expected = 0;
    while (c(s, "c2").ledger.current.month === startMonth) {
      students(s, 1_000);
      expected += (3 * 1_000) / 30;
      s = ok(step(s, TICK));
    }
    const paid = c(s, "c2").ledger.months.at(-1)?.costs.licenses;
    const got = c(s, "c1").ledger.months.at(-1)?.income.licenseIncome;
    // The tick that closes the month builds up nothing before it.
    expect(paid).toBe(Math.round(expected - 100));
    expect(got).toBe(paid);
    expect(s.licenses.deals[0]?.owed).toBeCloseTo(100, 6); // today's, for the new month
  });

  it("are $0 for a licensee with no customers in the licensed groups", () => {
    let s = licensed();
    const startMonth = c(s, "c2").ledger.current.month;
    while (c(s, "c2").ledger.current.month === startMonth) {
      students(s, 0);
      s = ok(step(s, TICK));
    }
    expect(c(s, "c2").ledger.months.at(-1)?.costs.licenses).toBe(0);
    expect(c(s, "c1").ledger.months.at(-1)?.income.licenseIncome).toBe(0);
  });

  it("are told in the news when a license starts", () => {
    const s = licensed(market3(), ["students", "everyday"]);
    expect(s.feed.at(-1)).toMatchObject({
      kind: "news",
      companyId: "c2",
      headline: "Beta licenses Acme's Pengu for its students and everyday users",
    });
  });
});

describe("a license ends", () => {
  it("when the licensed AI is retired, that day, with the royalty so far", () => {
    let s = licensed();
    students(s, 600);
    s = ok(step(s, TICK)); // $60 built up
    s = ok(step(s, cmd({ type: "setPlan", plan: "free", productId: null })));
    s = ok(step(s, cmd({ type: "setPlan", plan: "personal", productId: null })));
    s = ok(step(s, cmd({ type: "retireProduct", productId: "p1" })));
    const paid = c(s, "c2").ledger.current.costs.licenses;
    const got = c(s, "c1").ledger.current.income.licenseIncome;
    s = ok(step(s, TICK));
    expect(s.licenses.deals).toEqual([]);
    expect(activeLicense(s, "p2", "students")).toBeNull();
    expect(c(s, "c2").ledger.current.costs.licenses - paid).toBe(60);
    expect(c(s, "c1").ledger.current.income.licenseIncome - got).toBe(60);
    expect(s.feed.at(-1)).toMatchObject({ headline: "Beta's license of Pengu ends" });
  });

  it("at the end of its term", () => {
    let s = licensed();
    s = ticks(s, 89);
    expect(s.licenses.deals).toHaveLength(1);
    s = ticks(s, 1);
    expect(s.licenses.deals).toEqual([]);
  });

  it("early, by the licensee only, paying one month at today's customers", () => {
    let s = licensed();
    students(s, 600);
    s = ok(step(s, TICK)); // $60 built up
    const dealId = s.licenses.deals[0]?.id as string;
    expect(rejected(step(s, cmd({ type: "endLicense", dealId })))).toBe(
      "Only the company that licenses it can end it early.",
    );
    students(s, 500);
    const paid = c(s, "c2").ledger.current.costs.licenses;
    s = ok(step(s, cmd({ type: "endLicense", dealId }, undefined, "u2")));
    expect(s.licenses.deals).toEqual([]);
    expect(c(s, "c2").ledger.current.costs.licenses - paid).toBe(60 + 1_500);
  });
});

describe("licensing in random games", () => {
  const PLAYERS = ["u1", "u2", "u3"] as const;
  const PRODUCTS = ["p1", "p2", "p3"] as const;
  const VERSIONS = ["v1", "v2", "v3"] as const;
  const id = (kind: string) => fc.integer({ min: 1, max: 5 }).map((n) => `${kind}-${n}`);
  const groups = fc.constantFrom<GroupId[]>(
    ["students"],
    ["students"],
    ["students", "everyday"],
    ["everyday"],
  );
  const input: fc.Arbitrary<Input> = fc.oneof(
    { arbitrary: fc.constant(TICK), weight: 6 },
    fc
      .record({
        u: fc.constantFrom(...PLAYERS),
        versionId: fc.constantFrom(...VERSIONS),
        priceCents: fc.constantFrom(300, 10, 2000),
        months: fc.constantFrom(3, 6, 12),
        groups,
        exclusive: fc.boolean(),
      })
      .map(({ u, ...r }) => cmd({ type: "offerLicense", ...r }, undefined, u)),
    {
      arbitrary: fc
        .record({
          u: fc.constantFrom(...PLAYERS),
          offerId: id("offer"),
          productId: fc.constantFrom(...PRODUCTS),
          groups,
        })
        .map(({ u, ...r }) => cmd({ type: "takeLicense", ...r }, undefined, u)),
      weight: 4,
    },
    fc
      .record({
        u: fc.constantFrom(...PLAYERS),
        versionId: fc.constantFrom(...VERSIONS),
        productId: fc.constantFrom(...PRODUCTS),
        groups,
      })
      .map(({ u, ...r }) =>
        cmd({ type: "proposeLicense", priceCents: 500, months: 3, ...r }, undefined, u),
      ),
    fc
      .record({
        u: fc.constantFrom(...PLAYERS),
        proposalId: id("proposal"),
        accept: fc.boolean(),
      })
      .map(({ u, ...r }) => cmd({ type: "answerProposal", ...r }, undefined, u)),
    fc
      .record({ u: fc.constantFrom(...PLAYERS), dealId: id("deal") })
      .map(({ u, dealId }) => cmd({ type: "endLicense", dealId }, undefined, u)),
  );

  it("keep every company's money adding up, and royalties equal on both sides", () => {
    let deals = 0;
    let moved = 0;
    let made = 0;
    fc.assert(
      fc.property(fc.array(input, { minLength: 30, maxLength: 120 }), (inputs) => {
        let s = market3();
        for (const id of ["c2", "c3"]) {
          s.market.customers[id] = {
            pacific: { students: { free: 800 }, everyday: { free: 400 } },
          };
        }
        for (const i of inputs) {
          const r = step(s, i);
          if (r.ok) s = r.state;
        }
        deals += s.licenses.seq;
        let paid = 0;
        let got = 0;
        for (const company of Object.values(s.companies)) {
          const months = [...company.ledger.months, company.ledger.current];
          const sum = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);
          const net = months.reduce(
            (n, m) =>
              n +
              sum(m.income) -
              sum(m.costs) +
              m.financing.loans +
              m.financing.investors -
              m.financing.repaid,
            0,
          );
          expect(company.cash + company.fund.balance).toBe(company.startCash + net);
          paid += months.reduce((n, m) => n + m.costs.licenses, 0);
          got += months.reduce((n, m) => n + m.income.licenseIncome, 0);
        }
        expect(got).toBe(paid);
        moved += paid;
        made += s.feed.filter((x) => x.kind === "news" && x.topic === "license").length;
      }),
      { numRuns: 60, seed: 3 },
    );
    expect(deals).toBeGreaterThan(30); // licenses really happen in these games
    expect(made).toBeGreaterThan(5); // deals start and end
    expect(moved).toBeGreaterThan(0); // and royalties are paid
  }, 60_000);
});

describe("licensed answers to customers", () => {
  const REPLY = "Pengu: <b>2+2</b> = 4 <|end|> 😀";
  const batch = (s: WorldState) =>
    step(
      s,
      sys({
        type: "feedbackScored",
        productId: "p2",
        key: "w1~deal-2",
        results: [
          {
            group: "students",
            questionId: "q1",
            question: "What is 2 + 2?",
            reply: REPLY,
            score: 1,
            poweredBy: "Pengu",
          },
        ],
      }),
    );

  it("post as the licensee's product, powered by the licensed AI, quoting it byte for byte", () => {
    const s = ok(batch(licensed()));
    const post = s.feed.findLast((i) => i.kind === "post");
    expect(post).toMatchObject({
      kind: "post",
      companyId: "c2",
      productName: "Owl",
      reply: REPLY,
      poweredBy: "Pengu",
    });
    const seen = projectView(s, "u3").feed.findLast((i) => i.kind === "post");
    expect(seen).toMatchObject({ reply: REPLY, poweredBy: "Pengu", productName: "Owl" });
  });

  it("move the licensee's product's satisfaction, and aren't the licensee's loved replies", () => {
    const before = licensed();
    const s = ok(batch(before));
    expect(s.market.satisfaction.p2?.students ?? 0).toBeGreaterThan(
      before.market.satisfaction.p2?.students ?? 0.5,
    );
    expect(c(s, "c2").loved).toBe(c(before, "c2").loved);
  });

  it("refuse a label that isn't a short text", () => {
    const s = licensed();
    const bad = step(
      s,
      sys({
        type: "feedbackScored",
        productId: "p2",
        key: "w1",
        results: [
          { group: "students", questionId: "q", question: "?", reply: "x", score: 1, poweredBy: 7 },
        ],
      }),
    );
    expect(rejected(bad)).toBe("That feedback isn't valid.");
  });
});

describe("rivals and licenses", () => {
  const PRICES = { premium: 800, budget: 150, specialist: 500, consumer: 300 } as const;

  it("offer each new AI they put on sale at their style's price, for all six groups", () => {
    let s = withRivals();
    s = ticks(s, 30);
    const rivals = Object.values(s.companies).filter((x) => x.rival);
    for (const rival of rivals) {
      const live = Object.values(rival.products).filter((p) => p.status === "live");
      const offers = s.licenses.offers.filter((o) => o.licensorId === rival.id);
      if (live.length === 0) {
        expect(offers).toEqual([]);
        continue;
      }
      const newest = live.sort((a, b) => b.launchDay - a.launchDay)[0];
      expect(offers).toEqual([
        expect.objectContaining({
          versionId: newest?.versionId,
          aiName: newest?.name,
          priceCents: PRICES[rival.rival?.style ?? "premium"],
          months: 6,
          groups: ["students", "creators", "developers", "businesses", "everyday", "researchers"],
          exclusive: false,
        }),
      ]);
    }
    expect(s.licenses.offers.length).toBeGreaterThan(0);
  });

  /** A world with rivals on sale and Acme selling `knowledge`-strong Pengu. */
  function strongPlayer(knowledge: number): WorldState {
    const s = ticks(withRivals(), 120);
    selling(s, "c1", "v1", "p1", "Pengu", knowledge);
    return s;
  }

  it("ask once to license a player's AI that beats their focus skill by 10, and not again within 90 days", () => {
    let s = ok(step(strongPlayer(100), TICK));
    const asks = s.licenses.proposals.filter((p) => p.licensorId === "c1");
    // Rivals whose best in their focus skill is 10 or more below Pengu's.
    expect(asks.length).toBeGreaterThan(0);
    for (const p of asks) {
      const rival = c(s, p.licenseeId);
      expect(p).toMatchObject({ versionId: "v1", months: 6, groups: [rival.rival?.focus] });
      expect(rival.products[p.productId]?.status).toBe("live");
      // Its style's price, but never more than a tenth of its last month's revenue in royalties.
      expect(p.priceCents).toBeLessThanOrEqual(PRICES[rival.rival?.style ?? "premium"]);
      expect(p.priceCents).toBeGreaterThanOrEqual(10);
      const last = rival.ledger.months.at(-1);
      const revenue = last ? Object.values(last.income).reduce((a, b) => a + b, 0) : 0;
      let focus = 0;
      for (const byGroup of Object.values(s.market.customers[rival.id] ?? {})) {
        for (const n of Object.values(byGroup?.[rival.rival?.focus ?? "students"] ?? {})) {
          focus += n ?? 0;
        }
      }
      expect((p.priceCents / 100) * focus).toBeLessThanOrEqual(0.1 * revenue + 1e-6);
    }
    const asked = asks.map((p) => p.licenseeId);
    // Declined: no second ask within 90 days.
    for (const p of asks) {
      s = ok(step(s, cmd({ type: "answerProposal", proposalId: p.id, accept: false })));
    }
    s = ticks(s, 85);
    expect(s.licenses.proposals.filter((p) => asked.includes(p.licenseeId))).toEqual([]);
  });

  it("don't ask when a tenth of their revenue can't pay 10 cents a customer", () => {
    const poor = strongPlayer(100);
    for (const rival of Object.values(poor.companies)) {
      const last = rival.rival ? rival.ledger.months.at(-1) : undefined;
      if (!last) continue;
      for (const k of Object.keys(last.income)) last.income[k as keyof typeof last.income] = 0;
      last.income.personal = 100;
    }
    const s = ok(step(poor, TICK));
    expect(s.licenses.proposals).toEqual([]);
  });

  it("don't ask about an AI that isn't 10 better than theirs", () => {
    const weak = strongPlayer(0);
    const scores = c(weak, "c1").versions.v1?.scores ?? {};
    for (const k of Object.keys(scores)) scores[k] = 0;
    const s = ticks(weak, 1);
    expect(s.licenses.proposals).toEqual([]);
  });
});
