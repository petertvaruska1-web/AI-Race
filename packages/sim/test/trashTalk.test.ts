import { describe, expect, it } from "vitest";
import { type CompanyState, step, type WorldState } from "../src/index.ts";
import { cmd, ok, runningWorld, sys, ticks, withRivals } from "./pack.ts";

const founderLines = (s: WorldState) =>
  s.feed.filter(
    (f) => f.kind === "news" && f.topic === "trashTalk" && f.headline.includes("'s founder: \""),
  );

const archOf = (s: WorldState) => s.companies[s.rivals?.archId ?? ""] as CompanyState;

/** c1 sells "Spark" on every plan from `countdown` days on. */
function launchSpark(s0: WorldState, countdown = 3): WorldState {
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
    scores: { knowledge: 90, coding: 90 },
    overall: 50,
    personalityWords: [],
    traits: null,
    published: false,
    packKey: null,
  };
  return ok(
    step(
      s,
      cmd(
        {
          type: "launchProduct",
          versionId: "v1",
          name: "Spark",
          plans: ["free", "personal", "business"],
          countdownDays: countdown,
        },
        "p1",
      ),
    ),
  );
}

describe("the arch-rival's trash-talk", () => {
  it("brags when the arch-rival launches", () => {
    let s = withRivals("balance-test");
    const before = founderLines(s).length;
    s = ticks(s, 6);
    const arch = archOf(s);
    const lines = founderLines(s).slice(before);
    expect(lines.length).toBeGreaterThanOrEqual(1);
    const first = lines[0];
    expect(first?.kind === "news" && first.headline.startsWith(`${arch.name}'s founder: "`)).toBe(
      true,
    );
    expect(first?.companyId).toBe(arch.id);
  });

  it("comments on the player's launch and on passing it, as labelled game talk", () => {
    let s = ticks(withRivals("balance-test"), 10);
    s = launchSpark(s);
    const before = founderLines(s).length;
    s = ticks(s, 3);
    const lines = founderLines(s).slice(before);
    expect(lines).toHaveLength(1);
    expect(lines[0]?.kind === "news" && /Spark|Acme/.test(lines[0].headline)).toBe(true);

    // Passing the arch-rival on customers: told once.
    s = ticks(s, 5);
    const mine = s.market.customers.c1 ?? {};
    s.market.customers.c1 = { ...mine, pacific: { everyday: { free: 1e9 } } };
    const aboutAcme = (x: WorldState) =>
      founderLines(x).filter((f) => f.kind === "news" && f.headline.includes("Acme")).length;
    const n = aboutAcme(s);
    s = ticks(s, 1);
    expect(aboutAcme(s)).toBe(n + 1);
    expect(s.rivals?.archPassed).toBe(true);
    s = ticks(s, 5);
    expect(aboutAcme(s)).toBe(n + 1);
  });

  it("speaks at most once per gap, and never without an arch-rival", () => {
    let s = ticks(withRivals("balance-test"), 10);
    // Two launches on the same day: one line.
    s = launchSpark(s);
    const before = founderLines(s).length;
    s = ticks(s, 3);
    expect(founderLines(s).length - before).toBe(1);
    expect(archOf(s).rival?.lastTrashTalkDay).toBe(s.day);

    const alone = ticks(launchSpark(runningWorld("balance-test")), 5);
    expect(founderLines(alone)).toHaveLength(0);
  });

  it("notices a rescue", () => {
    let s = ticks(withRivals("balance-test"), 10);
    const c1 = s.companies.c1 as CompanyState;
    c1.cash = -10_000;
    c1.startCash -= 10_000 + 250_000;
    s = ticks(s, 31);
    expect((s.companies.c1 as CompanyState).rescue).not.toBeNull();
    (archOf(s).rival as { lastTrashTalkDay: number | null }).lastTrashTalkDay = null;
    const before = founderLines(s).length;
    s = ok(step(s, cmd({ type: "chooseRescue", kind: "loan" })));
    expect(founderLines(s).length).toBe(before + 1);
    expect(founderLines(s).at(-1)?.kind === "news" && founderLines(s).at(-1)).toBeTruthy();
  });

  it("grumbles when the player beats it in a battle", () => {
    let s = ticks(launchSpark(ticks(withRivals("balance-test"), 10)), 6);
    const arch = archOf(s);
    const target = Object.values(arch.products).find((p) => p.status === "live");
    if (!target) throw new Error("the arch-rival has nothing on sale");
    s = ok(
      step(s, cmd({ type: "startBattle", rivalProductId: target.id, skill: "knowledge" }, "b1")),
    );
    (archOf(s).rival as { lastTrashTalkDay: number | null }).lastTrashTalkDay = null;
    const before = founderLines(s).length;
    const rounds = [0, 1, 2].map((i) => ({
      prompt: `Q${i}`,
      mine: { reply: "yes", score: 1 },
      theirs: { reply: "no", score: 0 },
    }));
    s = ok(step(s, sys({ type: "battleResolved", battleId: "b1", rounds })));
    const lines = founderLines(s).slice(before);
    expect(lines).toHaveLength(1);
    expect(lines[0]?.kind === "news" && lines[0].headline.includes("Acme")).toBe(true);
  });

  it("says the same lines for the same seed", () => {
    const run = () => founderLines(ticks(launchSpark(ticks(withRivals("balance-test"), 10)), 20));
    expect(run()).toEqual(run());
    expect(sys).toBeDefined();
  });
});
