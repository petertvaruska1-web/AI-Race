import { describe, expect, it } from "vitest";
import {
  awarenessOf,
  type CompanyState,
  getBalance,
  projectView,
  step,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, ticks } from "./pack.ts";

const c1 = (s: WorldState) => s.companies.c1 as CompanyState;
const E = getBalance("balance-v2").economy;
/** A running world whose company can afford anything. */
function rich(): WorldState {
  const s = structuredClone(runningWorld());
  c1(s).cash = 5_000_000;
  c1(s).startCash = 5_000_000;
  return s;
}
const sponsor = (s: WorldState, kind = "stadium", region = "pacific", id = "sp1") =>
  step(s, cmd({ type: "sponsor", region, kind }, id));

describe("sponsorships", () => {
  it("charge their price and make the news on day one", () => {
    const s = rich();
    const after = ok(sponsor(s));
    const price = E.sponsorships.kinds.stadium.price;
    expect(c1(after).cash).toBe(c1(s).cash - price);
    expect(c1(after).ledger.current.costs.sponsorships).toBe(price);
    const days = E.sponsorships.kinds.stadium.days;
    expect(c1(after).sponsorships.sp1).toMatchObject({
      region: "pacific",
      kind: "stadium",
      startDay: s.day,
      endDay: s.day + days,
    });
    const news = after.feed.filter((f) => f.kind === "news" && f.topic === "sponsorship");
    expect(news[0]?.kind === "news" && news[0].headline).toBe(
      "Acme puts its name on a stadium in the Pacific",
    );
  });

  it("raise their groups' awareness in their region every day until they end", () => {
    const s = ok(sponsor(rich()));
    const groups = E.sponsorships.kinds.stadium.groups;
    const later = ticks(s, 10);
    for (const g of groups) {
      expect(awarenessOf(later.market, "c1", "pacific", g)).toBeGreaterThan(
        awarenessOf(s.market, "c1", "pacific", g),
      );
    }
    const other = ticks(rich(), 10);
    expect(awarenessOf(later.market, "c1", "pacific", groups[0] ?? "everyday")).toBeGreaterThan(
      awarenessOf(other.market, "c1", "pacific", groups[0] ?? "everyday"),
    );
  });

  it("one of a kind per region at a time, and only with the money", () => {
    const s = ok(sponsor(rich()));
    expect(rejected(sponsor(s, "stadium", "pacific", "sp2"))).toMatch(/already/);
    expect(sponsor(s, "festival", "pacific", "sp2").ok).toBe(true);
    const poor = structuredClone(rich());
    c1(poor).cash = 10;
    expect(rejected(sponsor(poor))).toMatch(/costs/);
  });

  it("show on the map to everyone while they run", () => {
    const s = ok(sponsor(rich()));
    const me = projectView(s, "u1").companies.find((c) => c.id === "c1");
    expect(me?.sponsorships).toEqual([{ region: "pacific", kind: "stadium" }]);
    const over = ticks(s, E.sponsorships.kinds.stadium.days + 1);
    expect(projectView(over, "u1").companies.find((c) => c.id === "c1")?.sponsorships).toEqual([]);
  });
});

describe("looks", () => {
  it("start with the free ones, worn", () => {
    expect(projectView(runningWorld(), "u1").me.company?.looks).toEqual({
      owned: ["orb:classic", "logo:still"],
      orb: "orb:classic",
      logo: "logo:still",
    });
  });

  it("are bought once, worn at once, and can be switched back for free", () => {
    const s = runningWorld();
    const price = E.looks["orb:nebula"] ?? 0;
    expect(price).toBeGreaterThan(0);
    const bought = ok(step(s, cmd({ type: "buyLook", lookId: "orb:nebula" })));
    expect(c1(bought).cash).toBe(c1(s).cash - price);
    expect(c1(bought).ledger.current.costs.looks).toBe(price);
    expect(c1(bought).looks.orb).toBe("orb:nebula");
    expect(rejected(step(bought, cmd({ type: "buyLook", lookId: "orb:nebula" })))).toMatch(
      /already/,
    );
    const back = ok(step(bought, cmd({ type: "useLook", lookId: "orb:classic" })));
    expect(c1(back).looks.orb).toBe("orb:classic");
    expect(c1(back).cash).toBe(c1(bought).cash);
  });

  it("refuse unknown or unowned looks, and buying without the money", () => {
    const s = runningWorld();
    expect(rejected(step(s, cmd({ type: "buyLook", lookId: "orb:disco" })))).toMatch(/no such/i);
    expect(rejected(step(s, cmd({ type: "useLook", lookId: "logo:orbit" })))).toMatch(
      /Buy it first/,
    );
    const poor = structuredClone(s);
    c1(poor).cash = 1;
    expect(rejected(step(poor, cmd({ type: "buyLook", lookId: "logo:orbit" })))).toMatch(/costs/);
  });

  it("show the logo's motion to everyone, the orb to its owner", () => {
    const s = ok(step(runningWorld(), cmd({ type: "buyLook", lookId: "logo:pulse" })));
    const v = projectView(s, "u1");
    expect(v.companies.find((c) => c.id === "c1")?.logoMotion).toBe("logo:pulse");
    expect(v.me.company?.shop.looks.find((l) => l.id === "orb:supernova")?.price).toBe(
      E.looks["orb:supernova"],
    );
  });
});
