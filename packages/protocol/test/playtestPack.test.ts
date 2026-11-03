import { describe, expect, it } from "vitest";
import { ClientMessage, Command, TrainingPlan } from "../src/index.ts";

const ok = (cmd: unknown) => Command.safeParse(cmd).success;
const V = "6fa459ea-ee8a-4ca4-894e-db77e160355e";
const PLAN = {
  mix: {
    web: 1,
    books: 1,
    educational: 0,
    conversations: 0,
    code: 0,
    reasoning: 0,
    facts: 0,
    creative: 0,
  },
  cleaning: "standard",
  removeDuplicates: false,
  factCheck: false,
  variety: "natural",
  brainSize: "s",
  attentionSpan: 128,
  length: "short",
  learningStyle: 0.5,
  review: 0,
  probePrompt: "Hi",
  parentVersionId: null,
};

describe("playtest pack commands", () => {
  it("parse showcases, sponsorships, looks, apps and the fund", () => {
    expect(
      ok({ type: "startShowcase", game: "bridge", difficulty: "easy", versionId: V, settings: {} }),
    ).toBe(true);
    expect(
      ok({ type: "startShowcase", game: "chess", difficulty: "easy", versionId: V, settings: {} }),
    ).toBe(false);
    expect(ok({ type: "sponsor", region: "pacific", kind: "stadium" })).toBe(true);
    expect(ok({ type: "sponsor", region: "pacific", kind: "moon" })).toBe(false);
    expect(ok({ type: "buyLook", lookId: "orb:nebula" })).toBe(true);
    expect(ok({ type: "useLook", lookId: "logo:pulse" })).toBe(true);
    expect(ok({ type: "fundApp", kind: "homework", productId: V })).toBe(true);
    expect(ok({ type: "fundApp", kind: "casino", productId: V })).toBe(false);
    expect(ok({ type: "closeApp", appId: V })).toBe(true);
    expect(ok({ type: "invest", amount: 50_000 })).toBe(true);
    expect(ok({ type: "withdraw", amount: 1 })).toBe(true);
    expect(ok({ type: "invest", amount: 0 })).toBe(false);
    expect(ok({ type: "withdraw", amount: 2.5 })).toBe(false);
  });

  it("let a launch have a keynote, and old launches parse without one", () => {
    const launch = {
      type: "launchProduct",
      versionId: V,
      name: "Spark",
      plans: ["free"],
      countdownDays: 3,
    };
    const parsed = Command.parse(launch);
    expect(parsed.type === "launchProduct" && parsed.keynote).toBeNull();
    expect(ok({ ...launch, keynote: { questionId: "knowledge-customer-0003" } })).toBe(true);
  });

  it("let a training plan add tutoring, customers' conversations and expert checks", () => {
    const old = TrainingPlan.parse(PLAN);
    expect(old.tutoring).toBeNull();
    expect(old.learnFromCustomers).toBe(false);
    const extra = {
      ...PLAN,
      cleaning: "expert",
      tutoring: { dataset: "code", packs: 3 },
      learnFromCustomers: true,
    };
    expect(TrainingPlan.safeParse(extra).success).toBe(true);
    expect(
      TrainingPlan.safeParse({ ...extra, tutoring: { dataset: "code", packs: 6 } }).success,
    ).toBe(false);
    expect(
      TrainingPlan.safeParse({ ...extra, tutoring: { dataset: "notebook", packs: 1 } }).success,
    ).toBe(false);
  });

  it("let the player start a minigame, alone or against a rival's AI", () => {
    const msg = {
      v: 1,
      type: "minigame",
      cid: "7a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d",
      worldId: V,
      game: "quiz",
      difficulty: "hard",
      settings: { topic: "science" },
      versionId: V,
      opponent: null,
    };
    expect(ClientMessage.safeParse(msg).success).toBe(true);
    expect(ClientMessage.safeParse({ ...msg, opponent: V }).success).toBe(true);
    expect(ClientMessage.safeParse({ ...msg, game: "story", opponent: V }).success).toBe(false);
  });
});
