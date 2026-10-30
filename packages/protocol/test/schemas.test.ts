import { describe, expect, it } from "vitest";
import {
  ClientMessage,
  COST_CATEGORIES,
  Command,
  cleanText,
  FeedItemView,
  GROUP_IDS,
  PLAN_IDS,
  REGION_IDS,
  ServerMessage,
  TrainingPlan,
  type WorldView,
  zId,
  zName,
} from "../src/index.ts";

export const PLAN = {
  mix: {
    web: 12,
    books: 12,
    educational: 12,
    conversations: 12,
    code: 12,
    reasoning: 12,
    facts: 12,
    creative: 12,
  },
  cleaning: "standard",
  removeDuplicates: false,
  factCheck: false,
  variety: "natural",
  brainSize: "tiny",
  attentionSpan: 256,
  length: "standard",
  learningStyle: 0.5,
  review: 0,
  probePrompt: "",
  parentVersionId: null,
} as const;

const CID = "4b0c1c1e-8d6a-4f43-9a55-6f1c8a2b9e10";

const zeros = <K extends string>(keys: readonly K[]) =>
  Object.fromEntries(keys.map((k) => [k, 0])) as Record<K, number>;

const POST = {
  id: "f1",
  kind: "post",
  day: 3,
  companyId: "c1",
  sentiment: 0.8,
  author: { name: "Maya K.", handle: "@maya_k42", group: "students", spotlightId: "spot-1" },
  productName: "Spark",
  question: "What is the capital of France?",
  reply: "  <b>Paris</b> <|end|>\u0007 ",
  tone: "praise",
} as const;

function sampleView(): WorldView {
  const shape = { nLayer: 4, dModel: 128, ctxLen: 256 };
  const month = {
    month: "2027-01",
    income: { personal: 120, business: 0 },
    costs: { ...zeros(COST_CATEGORIES), training: 42000, serving: 3 },
    revenueByGroup: { ...zeros(GROUP_IDS), students: 120 },
    revenueByPlan: { ...zeros(PLAN_IDS), personal: 120 },
    revenueByRegion: { ...zeros(REGION_IDS), pacific: 120 },
    customersEnd: 0,
    customersEndByGroup: zeros(GROUP_IDS),
    financing: { loans: 0, investors: 0, repaid: 0 },
  };
  return {
    world: {
      id: "w1",
      name: "Friday night",
      phase: "running",
      balance: "balance-v2",
      day: 3,
      date: "2027-01-04",
      speed: 1,
      paused: false,
      hostUserId: "u1",
      effectiveHostUserId: "u1",
      inviteCode: "ABCD-1234",
      market: {
        adoption: 0.1,
        adoptable: 33_500_000,
        regions: Object.fromEntries(REGION_IDS.map((r) => [r, { adoptable: 3_700_000 }])) as Record<
          (typeof REGION_IDS)[number],
          { adoptable: number }
        >,
      },
      rivals: "none",
    },
    members: [{ userId: "u1", displayName: "Ana", online: true, companyId: "c1" }],
    companies: [
      {
        id: "c1",
        name: "Acme",
        logo: { shape: "circle", symbol: "spark", initials: "AC" },
        palette: { primary: "#4f8cff", secondary: "#ffb020" },
        ownerUserId: "u1",
        rival: null,
        publishedVersions: [],
        products: [
          {
            id: "p1",
            name: "Spark",
            status: "live",
            launchDay: 2,
            version: {
              id: "v-1",
              label: "v1",
              createdDay: 0,
              scores: { language: 41.5 },
              overall: 20.1,
              personalityWords: ["curious"],
            },
          },
        ],
        prices: { free: true, personal: 10, business: null },
        customers: 1230,
        customersByRegion: Object.fromEntries(REGION_IDS.map((r) => [r, 5])) as Record<
          (typeof REGION_IDS)[number],
          number
        >,
        ownership: 0.75,
        reputation: 52,
        followers: 37,
      },
    ],
    feed: [
      POST,
      {
        id: "f2",
        kind: "news",
        day: 2,
        companyId: "c1",
        sentiment: 0.5,
        topic: "launch",
        headline: "Acme launches Spark, scoring 42 in language",
      },
    ],
    spotlights: [
      {
        id: "spot-1",
        name: "Maya K.",
        handle: "@maya_k42",
        group: "students",
        region: "pacific",
        bio: "Writes essays at 2 a.m.",
      },
    ],
    me: {
      userId: "u1",
      company: {
        id: "c1",
        cash: 208000,
        products: [
          {
            id: "p1",
            name: "Spark",
            versionId: "v-1",
            status: "live",
            announcedDay: 0,
            launchDay: 2,
            hype: 0.1,
          },
        ],
        plans: {
          free: { productId: "p1" },
          personal: { productId: "p1", price: 10 },
          business: { productId: null, price: 30 },
        },
        customers: {
          total: 1234,
          byGroup: Object.fromEntries(GROUP_IDS.map((g) => [g, zeros(PLAN_IDS)])) as Record<
            (typeof GROUP_IDS)[number],
            Record<(typeof PLAN_IDS)[number], number>
          >,
          byRegion: Object.fromEntries(REGION_IDS.map((r) => [r, zeros(GROUP_IDS)])) as Record<
            (typeof REGION_IDS)[number],
            Record<(typeof GROUP_IDS)[number], number>
          >,
        },
        satisfaction: {
          p1: { ...Object.fromEntries(GROUP_IDS.map((g) => [g, 0.5])), students: 0.62 } as Record<
            (typeof GROUP_IDS)[number],
            number
          >,
        },
        ledger: { current: month, months: [] },
        fame: { reputation: 52, followers: 37 },
        compute: {
          cloudPflops: 0.00025,
          ownedPflops: 0,
          trainingShare: 0,
          loadPflops: 0.0001,
          served: 1,
        },
        rescue: { deadlineDay: 45 },
        loan: { balance: 120_000, principalPerMonth: 5000, monthlyRate: 0.04 },
        reports: [
          {
            month: "2027-02",
            revenue: 1200,
            costs: 300,
            profit: 900,
            lastProfit: null,
            customersStart: 0,
            customersEnd: 140,
            growthByGroup: { ...zeros(GROUP_IDS), students: 140 },
            mood: 0.62,
            best: {
              author: { name: "Iris L.", handle: "@iris_l51", group: "students" },
              productName: "Spark",
              question: "Capital of France?",
              reply: "<b>Paris</b> <|end|>",
              tone: "praise",
            },
            worst: null,
            headline: "Acme launches Spark",
            grade: "B",
          },
        ],
        dataCenters: [{ id: "dc1", region: "pacific", startDay: 1, openDay: 61, open: false }],
        campaigns: [
          {
            id: "k1",
            region: "pacific",
            group: "all",
            budgetPerDay: 2000,
            startDay: 1,
            endDay: 31,
            spent: 4000,
          },
        ],
        history: {
          "3": {
            cash: 120_000,
            customers: Object.fromEntries(GROUP_IDS.map((g) => [g, 4])) as Record<
              (typeof GROUP_IDS)[number],
              number
            >,
            regions: zeros(REGION_IDS),
            paying: 6,
            followers: 37,
            reputation: 52,
          },
        },
        runs: [
          {
            id: "r1",
            status: "training",
            computeStarted: true,
            plan: { ...PLAN, mix: { ...PLAN.mix } },
            preview: {
              params: 1377408,
              tokens: 12582912,
              flops: 1.04e17,
              cost: 42000,
              durationDays: 13,
            },
            startDay: 1,
            endDay: 14,
            progress: 0.15,
            failure: null,
            versionId: null,
          },
        ],
        versions: [
          {
            id: "v-1",
            label: "v1",
            runId: "r0",
            parentId: null,
            shape,
            createdDay: 0,
            trainedTokens: 12582912,
            scores: { language: 41.5, coding: null },
            overall: 20.1,
            personalityWords: ["curious"],
            traits: { curious: 0.4 },
            published: false,
          },
        ],
      },
    },
  };
}

describe("cleanText", () => {
  it("strips C0 controls and DEL and trims, keeping other marks", () => {
    expect(cleanText("  Ac\u0007me‮ ", {})).toBe("Acme‮");
    expect(cleanText("a\u007fb\u0000c", {})).toBe("abc");
  });

  it("keeps newlines only when allowed", () => {
    expect(cleanText("one\ntwo", {})).toBe("onetwo");
    expect(cleanText("one\ntwo\r", { allowNewlines: true })).toBe("one\ntwo");
  });
});

describe("zName", () => {
  const name = zName(2, 32);

  it("rejects too short and too long names", () => {
    expect(name.safeParse("a").success).toBe(false);
    expect(name.safeParse("x".repeat(33)).success).toBe(false);
    expect(name.safeParse("  \u0007a ").success).toBe(false);
  });

  it("accepts markup as plain text and returns the cleaned name", () => {
    expect(name.parse("<b>Acme</b>")).toBe("<b>Acme</b>");
    expect(name.parse("  Acme\u0007 ")).toBe("Acme");
  });
});

describe("zId", () => {
  it("refuses names of built-in object parts, which a lookup would mistake for entries", () => {
    for (const id of ["__proto__", "constructor", "toString", "hasOwnProperty", "valueOf"]) {
      expect(zId.safeParse(id).success, id).toBe(false);
    }
    for (const id of ["p1", "proto", "my_constructor", "4b0c1c1e-8d6a-4f43-9a55-6f1c8a2b9e10"]) {
      expect(zId.safeParse(id).success, id).toBe(true);
    }
  });
});

describe("TrainingPlan", () => {
  it("accepts the default plan", () => {
    expect(TrainingPlan.safeParse(PLAN).success).toBe(true);
  });

  it.each([
    ["a mix weight above 100", { mix: { ...PLAN.mix, web: 101 } }],
    ["a fractional mix weight", { mix: { ...PLAN.mix, web: 1.5 } }],
    ["a missing dataset", { mix: { web: 100 } }],
    ["learning style above 1", { learningStyle: 1.5 }],
    ["review above 0.5", { review: 0.6 }],
    ["a probe prompt over 120 characters", { probePrompt: "x".repeat(121) }],
  ])("rejects %s", (_label, change) => {
    expect(TrainingPlan.safeParse({ ...PLAN, ...change }).success).toBe(false);
  });
});

describe("Command", () => {
  it("parses each command type", () => {
    const cmds = [
      {
        type: "createCompany",
        name: "Acme",
        logo: { shape: "hexagon", symbol: null, initials: "A1" },
        palette: { primary: "#4F8CFF", secondary: "#ffb020" },
      },
      { type: "startWorld" },
      { type: "setSpeed", speed: 4 },
      { type: "setPaused", paused: true },
      { type: "startTraining", plan: PLAN },
      { type: "publishVersion", versionId: "v-1", published: true },
    ];
    for (const cmd of cmds) expect(Command.safeParse(cmd).success, cmd.type).toBe(true);
  });

  it("rejects bad speeds, logos and palettes", () => {
    expect(Command.safeParse({ type: "setSpeed", speed: 3 }).success).toBe(false);
    const company = {
      type: "createCompany",
      name: "Acme",
      logo: { shape: "circle", symbol: "spark", initials: "AC" },
      palette: { primary: "#4f8cff", secondary: "#ffb020" },
    };
    expect(
      Command.safeParse({ ...company, logo: { ...company.logo, initials: "abc" } }).success,
    ).toBe(false);
    expect(
      Command.safeParse({ ...company, logo: { ...company.logo, symbol: "skull" } }).success,
    ).toBe(false);
    expect(
      Command.safeParse({ ...company, palette: { primary: "red", secondary: "#ffb020" } }).success,
    ).toBe(false);
  });
});

describe("ClientMessage", () => {
  it("requires a UUID command id", () => {
    expect(() =>
      ClientMessage.parse({
        v: 1,
        type: "command",
        cid: "not-a-uuid",
        worldId: "w1",
        cmd: { type: "startWorld" },
      }),
    ).toThrow();
    expect(
      ClientMessage.parse({
        v: 1,
        type: "command",
        cid: CID,
        worldId: "w1",
        cmd: { type: "startWorld" },
      }).type,
    ).toBe("command");
  });

  it("requires protocol version 1", () => {
    expect(ClientMessage.safeParse({ v: 2, type: "watch", worldId: "w1" }).success).toBe(false);
  });

  it("limits chat messages to 500 characters and keeps their newlines", () => {
    const chat = { v: 1, type: "chat", cid: CID, worldId: "w1", threadId: null, versionId: "v-1" };
    expect(ClientMessage.safeParse({ ...chat, text: "x".repeat(501) }).success).toBe(false);
    expect(ClientMessage.safeParse({ ...chat, text: " \u0007 " }).success).toBe(false);
    const parsed = ClientMessage.parse({ ...chat, text: "hi\nthere <|end|>" });
    expect(parsed.type === "chat" && parsed.text).toBe("hi\nthere <|end|>");
  });
});

describe("ServerMessage", () => {
  it("round-trips every message type through JSON", () => {
    const messages = [
      { v: 1, type: "welcome", userId: "u1" },
      { v: 1, type: "snapshot", worldId: "w1", version: 7, view: sampleView() },
      {
        v: 1,
        type: "patch",
        worldId: "w1",
        from: 7,
        to: 8,
        ops: [{ op: "set", path: ["world", "day"], value: 4 }],
      },
      { v: 1, type: "ack", cid: CID, duplicate: false },
      { v: 1, type: "ack", cid: CID, duplicate: true, data: { runId: "r1" } },
      { v: 1, type: "reject", cid: CID, reason: "Not enough cash." },
      { v: 1, type: "event", worldId: "w1", kind: "versionReady", text: "v1 is ready." },
      { v: 1, type: "stream", sub: "chat:w1:t1", data: { kind: "chunk", text: "Hel" } },
      { v: 1, type: "queryResult", id: "q1", ok: true, data: { pong: true } },
      { v: 1, type: "queryResult", id: "q2", ok: false, error: "Unknown query." },
      { v: 1, type: "error", reason: "Please sign in again." },
    ];
    for (const m of messages) {
      const parsed = ServerMessage.parse(JSON.parse(JSON.stringify(m)));
      expect(parsed, m.type).toEqual(m);
    }
  });

  it("refuses a public company entry with private fields", () => {
    const view = sampleView();
    for (const leak of [{ cash: 1 }, { ledger: view.me.company?.ledger }, { satisfaction: {} }]) {
      const leaky = { ...view, companies: [{ ...view.companies[0], ...leak }] };
      expect(
        ServerMessage.safeParse({ v: 1, type: "snapshot", worldId: "w1", version: 1, view: leaky })
          .success,
      ).toBe(false);
    }
  });
});

describe("FeedItemView", () => {
  it("keeps a post's reply exactly as the AI wrote it", () => {
    const parsed = FeedItemView.parse(JSON.parse(JSON.stringify(POST)));
    expect(parsed).toEqual(POST);
    expect(parsed.kind === "post" && parsed.reply).toBe("  <b>Paris</b> <|end|>\u0007 ");
  });

  it("refuses an unknown news topic or a sentiment out of range", () => {
    const news = { id: "f2", kind: "news", day: 1, companyId: null, sentiment: 0, headline: "x" };
    expect(FeedItemView.safeParse({ ...news, topic: "launch" }).success).toBe(true);
    expect(FeedItemView.safeParse({ ...news, topic: "gossip" }).success).toBe(false);
    expect(FeedItemView.safeParse({ ...POST, sentiment: 1.5 }).success).toBe(false);
  });
});
