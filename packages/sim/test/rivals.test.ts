import { ARCHETYPE_STYLE, RIVAL_STYLES } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import { getBalance, paletteProblem, projectView, step, type WorldState } from "../src/index.ts";
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
