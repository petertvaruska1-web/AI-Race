import {
  type FeedItemView,
  GROUP_IDS,
  type GroupId,
  type MonthLedgerView,
  PLAN_IDS,
  type PlanId,
  type PrivateCompany,
  type ProductView,
  type PublicCompany,
  type PublicProduct,
  type PublicVersion,
  REGION_IDS,
  type RunView,
  type SpotlightView,
  type TrainingPlan,
  type VersionView,
  type WorldView,
} from "@airace/protocol";
import { adoptionShare, getBalance } from "./balance.ts";
import { dateOfDay } from "./calendar.ts";
import { customersOf, totalCustomers } from "./market.ts";
import { liveProductFor } from "./products.ts";
import type {
  CompanyState,
  FeedItem,
  MonthLedger,
  ProductState,
  RunState,
  Spotlight,
  VersionState,
  WorldState,
} from "./state.ts";
import { runProgress } from "./training.ts";
import { effectiveHost } from "./world.ts";

/** Feed items a view carries; older ones stay in the world. */
const FEED_SHOWN = 50;
/** Closed months a view carries. */
const MONTHS_SHOWN = 12;

// The only path from world state to a client (R2, spec §6.4). Every view is
// built field by field: copying or filtering state objects would let a field
// added later leak by default.

/** What `viewerUserId` may see of the world. Throws for a non-member. */
export function projectView(state: WorldState, viewerUserId: string): WorldView {
  const viewer = state.members[viewerUserId];
  if (!viewer) throw new Error("not a member of this world");
  const own = viewer.companyId ? state.companies[viewer.companyId] : undefined;
  return {
    world: {
      id: state.worldId,
      name: state.name,
      phase: state.phase,
      balance: state.balance,
      day: state.day,
      date: dateOfDay(state.day),
      speed: state.speed,
      paused: state.paused,
      hostUserId: state.hostUserId,
      effectiveHostUserId: effectiveHost(state),
      inviteCode: state.inviteCode,
      market: marketView(state),
    },
    members: Object.values(state.members)
      .sort((a, b) => a.joinedSeq - b.joinedSeq)
      .map((m) => ({
        userId: m.userId,
        displayName: m.displayName,
        online: m.online,
        companyId: m.companyId,
      })),
    companies: sortedIds(state.companies).map((id) =>
      publicCompany(state, state.companies[id] as CompanyState),
    ),
    feed: state.feed.slice(-FEED_SHOWN).map(feedItemView),
    spotlights: state.spotlights.map(spotlightView),
    me: { userId: viewerUserId, company: own ? privateCompany(state, own) : null },
  };
}

function marketView(state: WorldState): WorldView["world"]["market"] {
  const e = getBalance(state.balance).economy;
  const adoption = adoptionShare(e, state.day);
  let people = 0;
  for (const region of REGION_IDS) people += e.population[region];
  return { adoption, adoptable: Math.round(people * adoption) };
}

function sortedIds(record: Record<string, unknown>): string[] {
  return Object.keys(record).sort();
}

function publicCompany(state: WorldState, c: CompanyState): PublicCompany {
  return {
    id: c.id,
    name: c.name,
    logo: { shape: c.logo.shape, symbol: c.logo.symbol, initials: c.logo.initials },
    palette: { primary: c.palette.primary, secondary: c.palette.secondary },
    ownerUserId: c.ownerUserId,
    publishedVersions: versionsInOrder(c)
      .filter((v) => v.published)
      .map(publicVersion),
    products: productsInOrder(c)
      .filter((p) => p.status !== "retired")
      .map((p) => publicProduct(c, p)),
    prices: {
      free: liveProductFor(c, "free") !== null,
      personal: liveProductFor(c, "personal") ? c.plans.personal.price : null,
      business: liveProductFor(c, "business") ? c.plans.business.price : null,
    },
    customers: threeFigures(totalCustomers(state.market, c.id)),
    reputation: Math.round(c.fame.reputation),
    followers: Math.round(c.fame.followers),
  };
}

// The AI behind a product stays private until launch day (R2: no unreleased versions).
function publicProduct(c: CompanyState, p: ProductState): PublicProduct {
  const version = p.status === "live" ? c.versions[p.versionId] : undefined;
  return {
    id: p.id,
    name: p.name,
    status: p.status === "live" ? "live" : "countdown",
    launchDay: p.launchDay,
    version: version ? publicVersion(version) : null,
  };
}

/** Whole customers, then three significant figures: rivals see the scale, not the detail. */
function threeFigures(n: number): number {
  const whole = Math.round(n);
  if (whole < 1000) return whole;
  const step = 10 ** (String(whole).length - 3);
  return Math.round(whole / step) * step;
}

function publicVersion(v: VersionState): PublicVersion {
  return {
    id: v.id,
    label: v.label,
    createdDay: v.createdDay,
    scores: v.scores ? { ...v.scores } : null,
    overall: v.overall,
    personalityWords: [...v.personalityWords],
  };
}

function privateCompany(state: WorldState, c: CompanyState): PrivateCompany {
  const runs = Object.values(c.runs).sort((a, b) => a.startDay - b.startDay || cmp(a.id, b.id));
  return {
    id: c.id,
    cash: c.cash,
    runs: runs.map((r) => runView(state, r)),
    versions: versionsInOrder(c).map(versionView),
    products: productsInOrder(c).map(productView),
    plans: {
      free: { productId: c.plans.free.productId },
      personal: { productId: c.plans.personal.productId, price: c.plans.personal.price },
      business: { productId: c.plans.business.productId, price: c.plans.business.price },
    },
    customers: customersView(state, c),
    satisfaction: satisfactionView(state, c),
    ledger: {
      current: monthView(c.ledger.current),
      months: c.ledger.months.slice(-MONTHS_SHOWN).map(monthView),
    },
    fame: { reputation: Math.round(c.fame.reputation), followers: Math.round(c.fame.followers) },
    history: Object.fromEntries(
      c.history.map(({ day, ...sample }) => [
        String(day),
        { ...sample, customers: { ...sample.customers } },
      ]),
    ),
  };
}

function productView(p: ProductState): ProductView {
  return {
    id: p.id,
    name: p.name,
    versionId: p.versionId,
    status: p.status,
    announcedDay: p.announcedDay,
    launchDay: p.launchDay,
    hype: p.hype,
  };
}

function customersView(state: WorldState, c: CompanyState): PrivateCompany["customers"] {
  const byGroup = {} as Record<GroupId, Record<PlanId, number>>;
  for (const group of GROUP_IDS) {
    const byPlan = {} as Record<PlanId, number>;
    for (const plan of PLAN_IDS) {
      let n = 0;
      for (const region of REGION_IDS) n += customersOf(state.market, c.id, region, group, plan);
      byPlan[plan] = Math.round(n);
    }
    byGroup[group] = byPlan;
  }
  return { total: Math.round(totalCustomers(state.market, c.id)), byGroup };
}

function satisfactionView(state: WorldState, c: CompanyState): PrivateCompany["satisfaction"] {
  const start = getBalance(state.balance).economy.feedback.satisfactionStart;
  const out: PrivateCompany["satisfaction"] = {};
  for (const p of productsInOrder(c)) {
    const byGroup = {} as Record<GroupId, number>;
    for (const group of GROUP_IDS) {
      byGroup[group] = state.market.satisfaction[p.id]?.[group] ?? start;
    }
    out[p.id] = byGroup;
  }
  return out;
}

function pick<K extends string>(keys: readonly K[], from: Record<K, number>): Record<K, number> {
  const out = {} as Record<K, number>;
  for (const k of keys) out[k] = from[k];
  return out;
}

function monthView(m: MonthLedger): MonthLedgerView {
  return {
    month: m.month,
    income: { personal: m.income.personal, business: m.income.business },
    costs: { training: m.costs.training, serving: m.costs.serving, dataChecks: m.costs.dataChecks },
    revenueByGroup: pick(GROUP_IDS, m.revenueByGroup),
    revenueByPlan: pick(PLAN_IDS, m.revenueByPlan),
    customersEnd: m.customersEnd,
  };
}

function feedItemView(i: FeedItem): FeedItemView {
  const base = { id: i.id, day: i.day, companyId: i.companyId, sentiment: i.sentiment };
  if (i.kind === "news") return { ...base, kind: "news", topic: i.topic, headline: i.headline };
  return {
    ...base,
    kind: "post",
    author: {
      name: i.author.name,
      handle: i.author.handle,
      group: i.author.group,
      spotlightId: i.author.spotlightId,
    },
    productName: i.productName,
    question: i.question,
    reply: i.reply,
    tone: i.tone,
  };
}

function spotlightView(p: Spotlight): SpotlightView {
  return { id: p.id, name: p.name, handle: p.handle, group: p.group, region: p.region, bio: p.bio };
}

// A held result stays out of the view: it is revealed when its game day comes (§7.3).
function runView(state: WorldState, r: RunState): RunView {
  return {
    id: r.id,
    status: r.status,
    computeStarted: r.computeStarted,
    plan: planView(r.plan),
    preview: {
      params: r.preview.params,
      tokens: r.preview.tokens,
      flops: r.preview.flops,
      cost: r.preview.cost,
      durationDays: r.preview.durationDays,
    },
    startDay: r.startDay,
    endDay: r.endDay,
    progress: r.status === "completed" ? 1 : runProgress(r, state.day),
    failure: r.failure,
    versionId: r.versionId,
  };
}

function planView(p: TrainingPlan): TrainingPlan {
  return {
    mix: { ...p.mix },
    cleaning: p.cleaning,
    removeDuplicates: p.removeDuplicates,
    factCheck: p.factCheck,
    variety: p.variety,
    brainSize: p.brainSize,
    attentionSpan: p.attentionSpan,
    length: p.length,
    learningStyle: p.learningStyle,
    review: p.review,
    probePrompt: p.probePrompt,
    parentVersionId: p.parentVersionId,
  };
}

function versionView(v: VersionState): VersionView {
  return {
    id: v.id,
    label: v.label,
    runId: v.runId,
    parentId: v.parentId,
    shape: { nLayer: v.nLayer, dModel: v.dModel, ctxLen: v.ctxLen },
    createdDay: v.createdDay,
    trainedTokens: v.trainedTokens,
    scores: v.scores ? { ...v.scores } : null,
    overall: v.overall,
    personalityWords: [...v.personalityWords],
    traits: v.traits ? { ...v.traits } : null,
    published: v.published,
  };
}

/** Products in the order they were announced. */
function productsInOrder(c: CompanyState): ProductState[] {
  return Object.values(c.products).sort(
    (a, b) => a.announcedDay - b.announcedDay || cmp(a.id, b.id),
  );
}

/** Versions oldest first (`v1`, `v2`, …). */
function versionsInOrder(c: CompanyState): VersionState[] {
  return Object.values(c.versions).sort((a, b) => labelNumber(a) - labelNumber(b));
}

function labelNumber(v: VersionState): number {
  return Number(v.label.slice(1));
}

function cmp(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
