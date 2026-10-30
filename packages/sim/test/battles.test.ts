import { WorldView } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  type CompanyState,
  dateOfDay,
  getBalance,
  projectView,
  stateHash,
  step,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys, ticks, withRivals } from "./pack.ts";

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
const FEE = getBalance("balance-test").economy.battles.fee;

/** c1 sells "Spark" (v1) on every plan. */
function launch(s0: WorldState, scores: Record<string, number | null> = SCORES): WorldState {
  const s = structuredClone(s0);
  const c1 = s.companies.c1 as CompanyState;
  c1.versions.v1 = {
    id: "v1",
    label: "v1",
    runId: "r1",
    parentId: null,
    nLayer: 4,
    dModel: 128,
    ctxLen: 256,
    createdDay: s.day,
    trainedTokens: 1,
    scores,
    overall: 40,
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
          plans: ["free", "personal", "business"],
          countdownDays: 3,
        },
        "p1",
      ),
    ),
  );
  return ticks(launched, 3);
}

/** A test-balance world where c1 and the arch-rival both have an AI on sale. */
function ready(scores?: Record<string, number | null>): { s: WorldState; target: string } {
  const s = ticks(launch(withRivals("balance-test"), scores), 8);
  const arch = s.companies["rival-1"] as CompanyState;
  const live = Object.values(arch.products).find((p) => p.status === "live");
  if (!live) throw new Error("the arch-rival has nothing on sale");
  return { s, target: live.id };
}

const start = (s: WorldState, target: string, skill = "knowledge", id = "b1") =>
  step(s, cmd({ type: "startBattle", rivalProductId: target, skill }, id));

const rounds = (mine: number[], theirs: number[]) =>
  mine.map((m, i) => ({
    prompt: `Q${i}?`,
    mine: { reply: `mine ${i}`, score: m },
    theirs: { reply: `theirs ${i}`, score: theirs[i] ?? 0 },
  }));

const resolve = (s: WorldState, battleId: string, r: unknown) =>
  step(s, sys({ type: "battleResolved", battleId, rounds: r }));

const battleOf = (s: WorldState, id = "b1") =>
  (s.companies.c1 as CompanyState).battles.find((b) => b.id === id);

describe("starting a battle", () => {
  it("refuses in plain words and changes nothing", () => {
    const alone = launch(runningWorld("balance-test"));
    expect(rejected(start(alone, "x"))).toBe("There are no rivals in this world.");
    const { s, target } = ready();
    expect(rejected(start(s, "nope"))).toBe("That AI isn't on sale.");
    expect(rejected(start(s, "p1"))).toBe("That AI isn't on sale.");
    const empty = ticks(withRivals("balance-test"), 10);
    expect(rejected(start(empty, target))).toBe("Launch an AI before battling.");
    const going = ok(start(s, target));
    expect(rejected(start(going, target, "coding", "b2"))).toBe("Your last battle is still going.");
    const done = ok(resolve(going, "b1", rounds([1, 1, 0], [0, 0, 1])));
    const cooldown = getBalance("balance-test").economy.battles.cooldownDays;
    const arch = s.companies["rival-1"]?.name;
    expect(rejected(start(done, target, "coding", "b2"))).toBe(
      `You battled ${arch} recently. Try again on ${dateOfDay(s.day + cooldown)}.`,
    );
    const noCreativity = ready({ ...SCORES, creativity: null });
    expect(rejected(start(noCreativity.s, noCreativity.target, "creativity"))).toBe(
      "Creativity can't be judged on this computer yet.",
    );
    const poor = structuredClone(s);
    (poor.companies.c1 as CompanyState).cash = 100;
    expect(rejected(start(poor, target))).toBe("A battle costs $5,000. You have $100.");
    expect(stateHash(s)).toBe(stateHash(ready().s));
  });

  it("books the fee once and puts the best AI for the skill in the ring", () => {
    const { s, target } = ready();
    const cash = (s.companies.c1 as CompanyState).cash;
    const after = ok(start(s, target));
    const c1 = after.companies.c1 as CompanyState;
    expect(c1.cash).toBe(cash - FEE);
    expect(c1.ledger.current.costs.marketing).toBe(FEE);
    expect(battleOf(after)).toMatchObject({
      status: "running",
      skill: "knowledge",
      rivalId: "rival-1",
      mine: { productId: "p1", versionId: "v1", productName: "Spark" },
      theirs: { productId: target },
    });
  });
});

describe("a battle's result", () => {
  it("a win moves ranking points by Elo, adds followers and tells one story", () => {
    const { s, target } = ready();
    const going = ok(start(s, target));
    const followers = (going.companies.c1 as CompanyState).fame.followers;
    const done = ok(resolve(going, "b1", rounds([1, 1, 0], [0, 0, 0.5])));
    expect(battleOf(done)).toMatchObject({ status: "done", outcome: "win" });
    expect(done.points.knowledge).toEqual({ c1: 1016, "rival-1": 984 });
    expect((done.companies.c1 as CompanyState).fame.followers).toBeCloseTo(followers * 1.05);
    const story = done.feed.filter((f) => f.kind === "news" && f.topic === "battle");
    expect(story).toHaveLength(1);
    const arch = done.companies["rival-1"] as CompanyState;
    const theirs = arch.products[target]?.name;
    expect(story[0]?.kind === "news" && story[0].headline).toBe(
      `Acme's Spark beats ${arch.name}'s ${theirs} in a knowledge battle, 2–1`,
    );
  });

  it("a draw inside the margin moves no points", () => {
    const { s, target } = ready();
    const done = ok(resolve(ok(start(s, target)), "b1", rounds([1, 0, 0.5], [0, 1, 0.4])));
    expect(battleOf(done)?.outcome).toBe("draw");
    expect(done.points.knowledge).toEqual({ c1: 1000, "rival-1": 1000 });
  });

  it("keeps replies exactly as the AIs wrote them", () => {
    const { s, target } = ready();
    const hostile = "<b>x</b> <|end|> 😀";
    const r = rounds([1, 0, 0], [0, 1, 1]);
    if (r[0]) r[0].mine.reply = hostile;
    const done = ok(resolve(ok(start(s, target)), "b1", r));
    expect(battleOf(done)?.rounds?.[0]?.mine.reply).toBe(hostile);
    expect(battleOf(done)?.outcome).toBe("loss");
  });

  it("a called-off battle refunds the fee once and changes no points", () => {
    const { s, target } = ready();
    const cash = (s.companies.c1 as CompanyState).cash;
    const going = ok(start(s, target));
    const off = ok(step(going, sys({ type: "battleResolved", battleId: "b1", failed: true })));
    expect((off.companies.c1 as CompanyState).cash).toBe(cash);
    expect((off.companies.c1 as CompanyState).ledger.current.costs.marketing).toBe(0);
    expect(battleOf(off)?.status).toBe("calledOff");
    expect(off.points).toEqual({});
    expect(rejected(resolve(off, "b1", rounds([1, 1, 1], [0, 0, 0])))).toBe("This battle is over.");
  });

  it("refuses a result with the wrong rounds or scores", () => {
    const { s, target } = ready();
    const going = ok(start(s, target));
    expect(rejected(resolve(going, "b1", rounds([1, 1], [0, 0])))).toBe(
      "That battle result isn't valid.",
    );
    expect(rejected(resolve(going, "b1", rounds([1, 2, 0], [0, 0, 0])))).toBe(
      "That battle result isn't valid.",
    );
    expect(rejected(resolve(going, "nope", rounds([1, 1, 0], [0, 0, 0])))).toBe(
      "This battle is over.",
    );
  });

  it("still applies when the rival's AI retired meanwhile", () => {
    const { s, target } = ready();
    const going = ok(start(s, target));
    const arch = going.companies["rival-1"] as CompanyState;
    const product = arch.products[target];
    if (product) product.status = "retired";
    const done = ok(resolve(going, "b1", rounds([1, 1, 0], [0, 0, 0])));
    expect(battleOf(done)?.outcome).toBe("win");
  });
});

describe("battles in the view", () => {
  it("shows your battles to you only, without the seed or the pack key", () => {
    const { s, target } = ready();
    const done = ok(resolve(ok(start(s, target)), "b1", rounds([1, 1, 0], [0, 0, 0])));
    const view = WorldView.parse(projectView(done, "u1"));
    expect(view.me.company?.battles).toHaveLength(1);
    const text = JSON.stringify(view);
    expect(text).not.toContain('"seed"');
    expect(text).not.toContain('"packKey"');
    expect(view.world.league.points.knowledge.map((r) => [r.companyId, r.points])).toEqual([
      ["c1", 1016],
      ["rival-1", 984],
    ]);
  });
});
