import { ARCHETYPE_STYLE, REGION_IDS, RIVAL_STYLES } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  adoptable,
  type CompanyState,
  getBalance,
  paletteProblem,
  projectView,
  step,
  totalCustomers,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, SYNTHETIC_PACK, sys, ticks, withRivals } from "./pack.ts";

const rivalsOf = (s: WorldState) =>
  Object.values(s.companies)
    .filter((c) => c.rival !== null)
    .sort((a, b) => a.id.localeCompare(b.id));

const join = (s: WorldState) => step(s, sys({ type: "rivalsJoined", pack: SYNTHETIC_PACK }));

describe("rivals join a world", () => {
  it("seats an arch-rival and three rivals, the same cast for the same seed", () => {
    const a = withRivals();
    const b = withRivals();
    const cast = (s: WorldState) =>
      rivalsOf(s).map((c) => ({
        id: c.id,
        name: c.name,
        style: c.rival?.style,
        archetype: c.rival?.archetype,
        lineage: c.rival?.lineageId,
        days: c.rival?.upcoming.map((u) => u.releaseDay),
      }));
    expect(cast(a)).toEqual(cast(b));
    expect(rivalsOf(a).map((c) => c.id)).toEqual(["rival-1", "rival-2", "rival-3", "rival-4"]);
    expect(a.rivals).toEqual({ joinedDay: a.day, archId: "rival-1" });
  });

  it("draws different archetypes across seeds, each with its own style", () => {
    const archetypes = new Set<string>();
    for (let seed = 1; seed <= 20; seed++) {
      const s = withRivals("balance-v2", seed);
      const arch = s.companies["rival-1"]?.rival;
      if (!arch?.archetype) throw new Error("no arch-rival");
      archetypes.add(arch.archetype);
      expect(arch.style).toBe(ARCHETYPE_STYLE[arch.archetype]);
      const others = rivalsOf(s).slice(1);
      expect(others.every((c) => c.rival?.archetype === null)).toBe(true);
      expect(new Set(others.map((c) => c.rival?.style)).size).toBe(3);
      expect(new Set(rivalsOf(s).map((c) => c.rival?.lineageId)).size).toBe(4);
    }
    expect(archetypes.size).toBeGreaterThanOrEqual(2);
  });

  it("gives unique names, never a player's, and distinct readable palettes", () => {
    for (let seed = 1; seed <= 10; seed++) {
      const s = withRivals("balance-v2", seed);
      const names = Object.values(s.companies).map((c) => c.name.toLowerCase());
      expect(new Set(names).size).toBe(names.length);
      const primaries = Object.values(s.companies).map((c) => c.palette.primary.toLowerCase());
      expect(new Set(primaries).size).toBe(primaries.length);
      for (const c of rivalsOf(s)) expect(paletteProblem(c.palette)).toBeNull();
    }
  });

  it("skips a name a player already uses", () => {
    const names = new Set<string>();
    for (let seed = 1; seed <= 20; seed++) {
      for (const c of rivalsOf(withRivals("balance-v2", seed))) names.add(c.name);
    }
    const taken = [...names][0] ?? "";
    let s = runningWorld();
    s.companies.c1 = { ...(s.companies.c1 as NonNullable<WorldState["companies"][string]>) };
    (s.companies.c1 as { name: string }).name = taken;
    for (let seed = 1; seed <= 20; seed++) {
      s = { ...s, rng: withRivals("balance-v2", seed).rng };
      const joined = ok(join(s));
      expect(rivalsOf(joined).some((c) => c.name.toLowerCase() === taken.toLowerCase())).toBe(
        false,
      );
    }
  });

  it("starts rivals with their style's cash and no owner", () => {
    const s = withRivals();
    const b = getBalance("balance-v2");
    for (const c of rivalsOf(s)) {
      const style = c.rival?.style ?? "budget";
      const arch = c.id === "rival-1" ? b.economy.rivals.archBonus : 1;
      expect(c.ownerUserId).toBeNull();
      expect(c.startCash).toBe(
        Math.round(b.foundingCash * b.economy.rivals.startCash[style] * arch),
      );
      expect(c.cash).toBe(c.startCash);
    }
    expect(s.companies.c1?.startCash).toBe(b.foundingCash);
  });

  it("schedules releases after joining, in order", () => {
    const s = withRivals();
    for (const c of rivalsOf(s)) {
      const days = c.rival?.upcoming.map((u) => u.releaseDay) ?? [];
      expect(days.length).toBe(3);
      expect(days[0]).toBeGreaterThanOrEqual(s.day + 1);
      for (let i = 1; i < days.length; i++) expect(days[i]).toBeGreaterThan(days[i - 1] ?? 0);
    }
  });

  it("joins a world later too, dating the releases from that day", () => {
    const late = ok(join(ticks(runningWorld(), 40)));
    expect(late.rivals?.joinedDay).toBe(40);
    for (const c of rivalsOf(late)) {
      expect(c.rival?.upcoming[0]?.releaseDay).toBeGreaterThanOrEqual(41);
    }
    const last = late.feed.filter((f) => f.kind === "news" && f.topic === "trashTalk").at(-1);
    expect(last?.kind === "news" && last.headline).toMatch(/^New rivals enter/);
    const early = withRivals();
    const first = early.feed.find((f) => f.kind === "news" && f.topic === "trashTalk");
    expect(first?.kind === "news" && first.headline).toMatch(/enters the AI race$/);
  });

  it("refuses a second join, a world in setup and an incomplete pack", () => {
    expect(rejected(join(withRivals()))).toBe("Rivals have already joined.");
    const setup = runningWorld();
    expect(rejected(join({ ...setup, phase: "setup" }))).toBe("The world is not running yet.");
    const broken = structuredClone(SYNTHETIC_PACK);
    const first = broken.lineages[0];
    if (first) first.style = "budget";
    expect(rejected(step(runningWorld(), sys({ type: "rivalsJoined", pack: broken })))).toBe(
      "The rival pack isn't valid.",
    );
  });

  it("shows rivals as companies without an owner, and says rivals joined", () => {
    const view = projectView(withRivals(), "u1");
    expect(view.world.rivals).toBe("joined");
    const rivals = view.companies.filter((c) => c.rival !== null);
    expect(rivals).toHaveLength(4);
    expect(rivals.every((c) => c.ownerUserId === null)).toBe(true);
    const arch = rivals.find((c) => c.id === "rival-1");
    expect(arch?.rival?.archetype).not.toBeNull();
    expect(rivals.filter((c) => c.rival?.archetype !== null)).toHaveLength(1);
    expect(RIVAL_STYLES).toContain(arch?.rival?.style);
    expect(projectView(runningWorld(), "u1").world.rivals).toBe("none");
  });

  it("never shows a rival's pack keys, lineage, upcoming releases or cash", () => {
    const s = ticks(withRivals(), 60);
    const json = JSON.stringify(projectView(s, "u1"));
    for (const l of SYNTHETIC_PACK.lineages) {
      expect(json).not.toContain(l.id);
      for (const v of l.versions) expect(json).not.toContain(v.key);
    }
    expect(json).not.toContain("upcoming");
    const view = projectView(s, "u1");
    for (const c of view.companies) expect(Object.keys(c)).not.toContain("cash");
    expect(view.me.company?.id).toBe("c1");
  });

  it("still lets the player act after rivals join", () => {
    const s = withRivals();
    expect(ok(step(s, cmd({ type: "setPaused", paused: true }))).paused).toBe(true);
  });
});

const sumCustomers = (s: WorldState, id: string) => totalCustomers(s.market, id);
const onSale = (c: CompanyState) => Object.values(c.products).filter((p) => p.status !== "retired");
const rivalById = (s: WorldState, id: string) => s.companies[id] as CompanyState;

describe("rivals run their businesses", () => {
  it("launch the arch-rival's first AI on schedule, with the pack's scores, and win customers", () => {
    let s = withRivals();
    const arch = rivalById(s, "rival-1");
    const first = arch.rival?.upcoming[0];
    if (!first) throw new Error("nothing scheduled");
    const countdown =
      getBalance("balance-v2").economy.rivals.countdownDays[arch.rival?.style ?? "premium"];
    s = ticks(s, first.releaseDay - s.day);
    const v = rivalById(s, "rival-1").versions["rival-1-v1"];
    expect(v?.packKey).toBe(first.key);
    expect(v?.scores).toEqual(first.scores);
    expect(v?.runId).toBe("pack");
    const p = rivalById(s, "rival-1").products["rival-1-p1"];
    expect(p?.status).toBe("countdown");
    expect(p?.launchDay).toBe(first.releaseDay + countdown);
    s = ticks(s, countdown);
    expect(rivalById(s, "rival-1").products["rival-1-p1"]?.status).toBe("live");
    s = ticks(s, 20);
    expect(sumCustomers(s, "rival-1")).toBeGreaterThan(0);
  });

  it("release every version within a year, never sell more than three, and keep prices in range", () => {
    let s = withRivals();
    const e = getBalance("balance-v2").economy;
    for (let d = 0; d < 365; d++) {
      s = ticks(s, 1);
      for (const c of rivalsOf(s)) {
        expect(onSale(c).length).toBeLessThanOrEqual(3);
        for (const plan of ["personal", "business"] as const) {
          const { min, max } = e.prices[plan];
          expect(c.plans[plan].price).toBeGreaterThanOrEqual(min);
          expect(c.plans[plan].price).toBeLessThanOrEqual(max);
        }
      }
    }
    for (const c of rivalsOf(s)) {
      expect(c.rival?.upcoming).toEqual([]);
      expect(c.rival?.released).toBe(3);
      expect(Number.isFinite(c.cash)).toBe(true);
    }
  });

  it("retires a product that powers no plan the day after", () => {
    let s = withRivals("balance-test");
    let idle: { company: string; product: string }[] = [];
    for (let d = 0; d < 60; d++) {
      s = ticks(s, 1);
      // Yesterday's idle AIs are gone today.
      for (const x of idle)
        expect(s.companies[x.company]?.products[x.product]?.status).toBe("retired");
      idle = [];
      for (const c of rivalsOf(s)) {
        const powered = new Set(
          (["free", "personal", "business"] as const).map((p) => c.plans[p].productId),
        );
        for (const p of Object.values(c.products)) {
          if (p.status === "live" && !powered.has(p.id))
            idle.push({ company: c.id, product: p.id });
        }
      }
    }
    const retired = rivalsOf(s).flatMap((c) =>
      Object.values(c.products).filter((p) => p.status === "retired"),
    );
    expect(retired.length).toBeGreaterThan(0);
  });

  it("take the emergency loan the day cash runs out", () => {
    let s = withRivals();
    const r = rivalById(s, "rival-2");
    r.cash = -1_000_000;
    r.startCash -= 1_000_000 + r.startCash; // keep the money identity: start below zero
    s = ticks(s, 31);
    const after = rivalById(s, "rival-2");
    expect(after.rescue).toBeNull();
    expect(after.loan?.balance).toBeGreaterThan(0);
    expect(after.cash).toBeGreaterThanOrEqual(0);
  });

  it("advertise to their focus group where most of them live, and build a data center when serving is costly", () => {
    let s = withRivals();
    const id = "rival-1";
    const c = rivalById(s, id);
    c.cash = 50_000_000;
    s = ticks(s, 120);
    const after = rivalById(s, id);
    const e = getBalance("balance-v2").economy;
    const focus = after.rival?.focus ?? "everyday";
    const campaigns = Object.values(after.campaigns);
    const started = after.ledger.months.some((m) => m.costs.marketing > 0) || campaigns.length > 0;
    expect(started).toBe(true);
    const best = REGION_IDS.reduce((a, b) =>
      adoptable(e, s.day, b, focus) > adoptable(e, s.day, a, focus) ? b : a,
    );
    for (const k of campaigns) {
      expect(k.group).toBe(focus);
      expect(k.region).toBe(best);
    }
    // A big serving bill and plenty of cash bring a data center at the next review.
    const before = Object.keys(after.dataCenters).length;
    const last = after.ledger.months.at(-1);
    if (last) last.costs.serving = e.rivals.dataCenterAtServingPerMonth;
    after.cash = 50_000_000;
    s = ticks(s, e.rivals.reviewEveryDays);
    expect(Object.keys(rivalById(s, id).dataCenters).length).toBeGreaterThan(before);
  });
});

describe("rivals over the years", { timeout: 60_000 }, () => {
  it("never go below zero customers, keep cash finite and prices in range, whatever the seed", () => {
    const e = getBalance("balance-v2").economy;
    for (const seed of [2, 3, 4]) {
      let s = withRivals("balance-v2", seed);
      for (let d = 0; d < 400; d += 10) {
        s = ticks(s, 10);
        for (const c of Object.values(s.companies)) {
          expect(Number.isFinite(c.cash)).toBe(true);
          for (const byRegion of Object.values(s.market.customers[c.id] ?? {})) {
            for (const byGroup of Object.values(byRegion ?? {})) {
              for (const n of Object.values(byGroup ?? {})) expect(n).toBeGreaterThanOrEqual(0);
            }
          }
          for (const plan of ["personal", "business"] as const) {
            expect(c.plans[plan].price).toBeGreaterThanOrEqual(e.prices[plan].min);
            expect(c.plans[plan].price).toBeLessThanOrEqual(e.prices[plan].max);
          }
        }
      }
    }
  });
});

describe("rival customer posts", () => {
  const batch = (productId: string, key: string) =>
    sys({
      type: "feedbackScored",
      productId,
      key,
      results: [
        { group: "students", questionId: "q1", question: "Q1?", reply: "fine", score: 0.6 },
        { group: "creators", questionId: "q2", question: "Q2?", reply: "<b>bad</b>", score: 0.05 },
        { group: "everyday", questionId: "q3", question: "Q3?", reply: "great", score: 0.9 },
      ],
    });

  it("make one post per batch, the most extreme reply, and never go viral", () => {
    let s = ticks(withRivals("balance-test"), 10);
    const product = Object.values(rivalById(s, "rival-1").products).find(
      (p) => p.status === "live",
    );
    if (!product) throw new Error("no live rival product");
    rivalById(s, "rival-1").fame.followers = 1_000_000;
    const before = s.feed.length;
    s = ok(step(s, batch(product.id, "w99")));
    const added = s.feed.slice(before);
    expect(added).toHaveLength(1);
    const post = added[0];
    expect(post?.kind === "post" && post.reply).toBe("<b>bad</b>");
  });
});
