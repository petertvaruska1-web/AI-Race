import {
  ARCHETYPE_STYLE,
  ARCHETYPE_WORDS,
  ARCHETYPES,
  type Archetype,
  REGION_IDS,
  RIVAL_STYLES,
  RivalPack,
  type RivalPackLineage,
  type RivalStyle,
} from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { campaignCommand } from "./campaigns.ts";
import { computeCommand, newCompute } from "./compute.ts";
import { tellNews } from "./feed.ts";
import { newLedger } from "./ledger.ts";
import { adoptable, customersByRegion, totalCustomers } from "./market.ts";
import { newCompanyBusiness, productCommand } from "./products.ts";
import { rescueCommand } from "./rescue.ts";
import { drawLook, rivalProductName } from "./rivalCast.ts";
import { rngInt } from "./rng.ts";
import type { CompanyState, PackRelease, RivalState, WorldState } from "./state.ts";

// Rivals (game-layer design §9): ordinary companies without an owner, so the
// market, money, fame and computers treat them like the player. They join from
// the rival pack's measurements, logged once as `rivalsJoined`.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

/** Rivals that join after this day announce themselves as newcomers. */
const EARLY_DAYS = 7;

export function isRival(company: CompanyState): boolean {
  return company.rival !== null;
}

/** Seats an arch-rival and three rivals from `pack`. Mutates `s`. */
export function joinRivals(s: WorldState, raw: unknown): Handled {
  if (s.phase !== "running") return no("The world is not running yet.");
  if (s.rivals !== null) return no("Rivals have already joined.");
  const parsed = RivalPack.safeParse(raw);
  if (!parsed.success) return no("The rival pack isn't valid.");
  const pack = parsed.data;
  const b = getBalance(s.balance);
  const r = b.economy.rivals;
  const draw = (n: number) => {
    const x = rngInt(s.rng, n);
    s.rng = x.next;
    return x.value;
  };
  const ofStyle = (style: RivalStyle) => pack.lineages.filter((l) => l.style === style);

  const archetype = ARCHETYPES[draw(ARCHETYPES.length)] as Archetype;
  const archStyle = ARCHETYPE_STYLE[archetype];
  const archLineage = ofStyle(archStyle)[draw(2)] as RivalPackLineage;
  const dropped = RIVAL_STYLES[draw(RIVAL_STYLES.length)];
  const cast: { lineage: RivalPackLineage; archetype: Archetype | null }[] = [
    { lineage: archLineage, archetype },
  ];
  for (const style of RIVAL_STYLES) {
    if (style === dropped) continue;
    const free = ofStyle(style).filter((l) => l.id !== archLineage.id);
    cast.push({ lineage: free[draw(free.length)] as RivalPackLineage, archetype: null });
  }

  const taken = new Set(Object.values(s.companies).map((c) => c.name.toLowerCase()));
  const primaries = new Set(Object.values(s.companies).map((c) => c.palette.primary.toLowerCase()));
  const ids: string[] = [];
  cast.forEach(({ lineage, archetype: type }, i) => {
    const drawn = drawLook(s.rng, type ?? lineage.style, taken, primaries);
    s.rng = drawn.rng;
    const id = `rival-${i + 1}`;
    const upcoming: PackRelease[] = [];
    let last = s.day;
    lineage.versions.forEach((v, k) => {
      const jitter = r.jitterDays > 0 ? draw(2 * r.jitterDays + 1) - r.jitterDays : 0;
      const planned = s.day + (r.firstRelease[i] ?? 0) + k * r.releaseGapDays + jitter;
      last = Math.max(last + 1, planned);
      upcoming.push({ ...v, releaseDay: last });
    });
    const cash = Math.round(b.foundingCash * r.startCash[lineage.style] * (type ? r.archBonus : 1));
    const company: CompanyState = {
      id,
      ownerUserId: null,
      startCash: cash,
      rival: {
        lineageId: lineage.id,
        style: lineage.style,
        focus: lineage.focus,
        archetype: type,
        productWord: drawn.look.productWord,
        upcoming,
        released: 0,
        lastReviewDay: s.day,
        lastTrashTalkDay: null,
        counters: { campaigns: 0, dataCenters: 0 },
      },
      lastRanks: null,
      name: drawn.look.name,
      logo: drawn.look.logo,
      palette: drawn.look.palette,
      cash,
      runs: {},
      versions: {},
      versionSeq: 0,
      ...newCompanyBusiness(b),
      ledger: newLedger(s.day),
      history: [],
      campaigns: {},
      compute: newCompute(b),
      dataCenters: {},
      reports: [],
      rescue: null,
      loan: null,
      ownership: 1,
    };
    s.companies[id] = company;
    rivalPrices(s, company);
    ids.push(id);
  });

  const arch = s.companies[ids[0] as string] as CompanyState;
  s.rivals = { joinedDay: s.day, archId: arch.id };
  const headline =
    s.day <= EARLY_DAYS
      ? `${arch.name}, ${ARCHETYPE_WORDS[archetype]}, enters the AI race`
      : `New rivals enter the AI race, led by ${arch.name}`;
  tellNews(s, arch, "trashTalk", headline, 0);
  return OK;
}

const PLANS = ["free", "personal", "business"] as const;

function clampPrice(s: WorldState, plan: "personal" | "business", price: number): number {
  const { min, max } = getBalance(s.balance).economy.prices[plan];
  return Math.min(max, Math.max(min, Math.round(price)));
}

/** A rival's starting prices: its style's share of each plan's default. */
export function rivalPrices(s: WorldState, company: CompanyState): void {
  const { prices, rivals } = getBalance(s.balance).economy;
  const factor = rivals.priceFactor[(company.rival as RivalState).style];
  company.plans.personal.price = clampPrice(s, "personal", prices.personal.default * factor);
  company.plans.business.price = clampPrice(s, "business", prices.business.default * factor);
}

function powersPlan(company: CompanyState, productId: string): boolean {
  return PLANS.some((plan) => company.plans[plan].productId === productId);
}

/** Puts released pack versions on sale; a release waits while three AIs fill every plan. */
function release(s: WorldState, company: CompanyState, rival: RivalState): void {
  const e = getBalance(s.balance).economy;
  while (rival.upcoming[0] && rival.upcoming[0].releaseDay <= s.day) {
    const onSale = Object.values(company.products).filter((p) => p.status !== "retired");
    if (onSale.length >= e.maxProducts) {
      const spare = onSale
        .filter((p) => p.status === "live" && !powersPlan(company, p.id))
        .sort((a, b) => a.launchDay - b.launchDay)[0];
      if (!spare) return;
      spare.status = "retired";
    }
    const next = rival.upcoming.shift() as PackRelease;
    const k = rival.released + 1;
    const id = `${company.id}-v${k}`;
    company.versions[id] = {
      id,
      label: `v${k}`,
      runId: "pack",
      parentId: k > 1 ? `${company.id}-v${k - 1}` : null,
      nLayer: next.nLayer,
      dModel: next.dModel,
      ctxLen: next.ctxLen,
      createdDay: s.day,
      trainedTokens: next.trainedTokens,
      scores: { ...next.scores },
      overall: next.overall,
      personalityWords: [...next.personalityWords],
      traits: { ...next.traits },
      published: false,
      packKey: next.key,
    };
    company.versionSeq = k;
    rival.released = k;
    productCommand(
      s,
      company,
      {
        type: "launchProduct",
        versionId: id,
        name: rivalProductName(rival.style, rival.productWord, k),
        plans: [...e.rivals.plans[rival.style]],
        countdownDays: e.rivals.countdownDays[rival.style],
      },
      `${company.id}-p${k}`,
    );
  }
}

/** Retires live AIs that power no plan once a newer one has taken their plans. */
function tidy(company: CompanyState, day: number): void {
  for (const p of Object.values(company.products)) {
    if (p.status === "live" && p.launchDay < day && !powersPlan(company, p.id)) {
      p.status = "retired";
    }
  }
}

function customersOn(company: CompanyState, day: number): number | null {
  const sample = company.history.find((h) => h.day === day);
  return sample ? Object.values(sample.customers).reduce((a, b) => a + b, 0) : null;
}

/** Every review: prices follow customers and load, then ads and data centers when cash allows. */
function review(s: WorldState, company: CompanyState, rival: RivalState): void {
  const e = getBalance(s.balance).economy;
  const r = e.rivals;
  rival.lastReviewDay = s.day;
  const now = totalCustomers(s.market, company.id);
  const before = customersOn(company, s.day - r.reviewEveryDays);
  let factor = 1;
  if (before !== null && now < before) factor *= 1 - r.priceStep;
  if (company.compute.served < 0.95) factor *= 1 + r.priceStep;
  if (factor !== 1) {
    for (const plan of ["personal", "business"] as const) {
      company.plans[plan].price = clampPrice(s, plan, company.plans[plan].price * factor);
    }
  }

  const selling = Object.values(company.products).some((p) => p.status !== "retired");
  if (selling && Object.keys(company.campaigns).length === 0 && company.cash >= r.ad.minCash) {
    const budgetPerDay = Math.min(
      r.ad.maxPerDay[rival.style],
      Math.floor(company.cash * r.ad.shareOfCash),
    );
    if (budgetPerDay >= e.campaigns.budget.min) {
      const region = REGION_IDS.reduce((a, b) =>
        adoptable(e, s.day, b, rival.focus) > adoptable(e, s.day, a, rival.focus) ? b : a,
      );
      rival.counters.campaigns += 1;
      campaignCommand(
        s,
        company,
        { type: "startCampaign", region, group: rival.focus, budgetPerDay, days: r.ad.days },
        `${company.id}-c${rival.counters.campaigns}`,
      );
    }
  }

  const serving = company.ledger.months.at(-1)?.costs.serving ?? 0;
  const centers = Object.keys(company.dataCenters).length;
  if (
    serving >= r.dataCenterAtServingPerMonth &&
    company.cash >= 2 * e.dataCenter.cost &&
    centers < e.dataCenter.maxPerCompany
  ) {
    const byRegion = customersByRegion(s.market, company.id);
    const region = REGION_IDS.reduce((a, b) => ((byRegion[b] ?? 0) > (byRegion[a] ?? 0) ? b : a));
    rival.counters.dataCenters += 1;
    computeCommand(
      s,
      company,
      { type: "buildDataCenter", region },
      `${company.id}-d${rival.counters.dataCenters}`,
    );
  }
}

/** The daily rival step, right after training settles: releases, lineup, reviews, rescues. */
export function rivalsDay(s: WorldState): void {
  if (!s.rivals) return;
  const every = getBalance(s.balance).economy.rivals.reviewEveryDays;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    const rival = company?.rival;
    if (!company || !rival) continue;
    release(s, company, rival);
    tidy(company, s.day);
    if (s.day - rival.lastReviewDay >= every) review(s, company, rival);
    if (company.rescue) rescueCommand(s, company, { type: "chooseRescue", kind: "loan" });
  }
}
