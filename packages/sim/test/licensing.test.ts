import type { GroupId } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  activeLicense,
  type CompanyState,
  projectView,
  step,
  upgradeState,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys } from "./pack.ts";

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
