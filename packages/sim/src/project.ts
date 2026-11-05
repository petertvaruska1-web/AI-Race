import {
  APP_KINDS,
  BATTLE_SKILLS,
  COST_CATEGORIES,
  type FeedItemView,
  GROUP_IDS,
  type GroupId,
  INCOME_CATEGORIES,
  LEAGUE_TABLES,
  type MonthLedgerView,
  PLAN_IDS,
  type PlanId,
  type PrivateCompany,
  type ProductView,
  type PublicCompany,
  type PublicProduct,
  type PublicVersion,
  REGION_IDS,
  type RegionId,
  type RunView,
  type ShopView,
  SPONSOR_KINDS,
  type SpotlightView,
  type TrainingPlan,
  type VersionView,
  type WorldView,
} from "@airace/protocol";
import { adoptionShare, type Balance, getBalance } from "./balance.ts";
import { dateOfDay } from "./calendar.ts";
import { ownedPflops } from "./compute.ts";
import { goalProgress } from "./goal.ts";
import { leagueTables } from "./league.ts";
import { customersByRegion, customersOf, totalCustomers } from "./market.ts";
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
function goalView(state: WorldState): WorldView["world"]["goal"] {
  const { spec, deadlineDay, reachedDay, missedDay, reachedBy } = state.goal;
  const goal = spec.goal;
  if (goal.kind === "none") return null;
  const target = "target" in goal ? goal.target : null;
  return { kind: goal.kind, target, deadlineDay, reachedDay, missedDay, reachedBy };
}

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
      rivals: state.rivals ? "joined" : "none",
      goal: goalView(state),
      fairMode: state.fairMode,
      league: leagueView(state, viewer.companyId),
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
  const regions = {} as WorldView["world"]["market"]["regions"];
  for (const region of REGION_IDS) {
    people += e.population[region];
    regions[region] = { adoptable: Math.round(e.population[region] * adoption) };
  }
  return { adoption, adoptable: Math.round(people * adoption), regions };
}

function leagueView(
  state: WorldState,
  viewerCompanyId: string | null,
): WorldView["world"]["league"] {
  const tables = leagueTables(state);
  const out = {} as WorldView["world"]["league"]["tables"];
  for (const table of LEAGUE_TABLES) {
    out[table] = tables[table].map((row, i) => {
      const own = row.companyId === viewerCompanyId;
      let value = row.value;
      // Others' revenue is a place only (R2); others' customers show at three figures.
      if (table === "revenue" && !own) value = null;
      else if (table === "customers" && value !== null) {
        value = own ? Math.round(value) : threeFigures(value);
      }
      return { companyId: row.companyId, rank: i + 1, value };
    });
  }
  const points = {} as WorldView["world"]["league"]["points"];
  for (const skill of BATTLE_SKILLS) {
    const table = state.points[skill] ?? {};
    points[skill] = Object.entries(table)
      .sort(([a, x], [b, y]) => y - x || a.localeCompare(b))
      .map(([companyId, p], i) => ({ companyId, rank: i + 1, points: p }));
  }
  return { tables: out, points };
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
    logoMotion: c.looks.logo,
    sponsorships: Object.values(c.sponsorships)
      .filter((x) => state.day >= x.startDay && state.day < x.endDay)
      .sort((a, b) => a.startDay - b.startDay || (a.id < b.id ? -1 : 1))
      .map((x) => ({ region: x.region, kind: x.kind })),
    ownerUserId: c.ownerUserId,
    rival: c.rival ? { style: c.rival.style, archetype: c.rival.archetype } : null,
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
    customersByRegion: perRegion(customersByRegion(state.market, c.id), threeFigures),
    ownership: c.ownership,
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
function perRegion(
  values: Record<RegionId, number>,
  round: (n: number) => number,
): Record<RegionId, number> {
  return Object.fromEntries(REGION_IDS.map((r) => [r, round(values[r])])) as Record<
    RegionId,
    number
  >;
}

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
    compute: {
      cloudPflops: c.compute.cloudPflops,
      ownedPflops: ownedPflops(c, state.day),
      trainingShare: c.compute.trainingShare,
      loadPflops: c.compute.loadPflops,
      served: c.compute.served,
    },
    rescue: c.rescue ? { deadlineDay: c.rescue.deadlineDay } : null,
    loan: c.loan
      ? {
          balance: c.loan.balance,
          principalPerMonth: c.loan.principalPerMonth,
          monthlyRate: getBalance(state.balance).economy.rescue.loanMonthlyRate,
        }
      : null,
    // The seed and the rival's pack key stay in the sim (R2).
    battles: c.battles.map((x) => ({
      id: x.id,
      day: x.day,
      skill: x.skill,
      rivalId: x.rivalId,
      mine: { ...x.mine },
      theirs: {
        productId: x.theirs.productId,
        versionId: x.theirs.versionId,
        productName: x.theirs.productName,
      },
      status: x.status,
      rounds: x.rounds
        ? x.rounds.map((r) => ({
            prompt: r.prompt,
            expected: r.expected ?? null,
            mine: { ...r.mine },
            theirs: { ...r.theirs },
            votes: r.votes ? { ...r.votes } : null,
          }))
        : null,
      outcome: x.outcome,
      pointsChange: x.pointsChange,
      opponentKind: x.opponentKind ?? "rival",
      defending: x.defending === true,
    })),
    looks: { owned: [...c.looks.owned], orb: c.looks.orb, logo: c.looks.logo },
    sponsorships: Object.values(c.sponsorships)
      .sort((a, b) => a.startDay - b.startDay || (a.id < b.id ? -1 : 1))
      .map((x) => ({ ...x })),
    apps: Object.values(c.apps)
      .sort((a, b) => a.startDay - b.startDay || (a.id < b.id ? -1 : 1))
      .map((a) => ({
        id: a.id,
        kind: a.kind,
        productId: a.productId,
        productName: c.products[a.productId]?.name ?? "",
        startDay: a.startDay,
        users: Math.round(a.users),
        rating: Math.round(a.rating * 10) / 10,
        status: a.status,
        reviews: a.reviews.map((r) => ({
          day: r.day,
          prompt: r.prompt,
          reply: r.reply,
          stars: r.stars,
        })),
      })),
    loved: c.loved,
    lessons: c.lessons ?? 0,
    notebook: c.notebook ? { words: c.notebook.words, savedDay: c.notebook.savedDay } : null,
    milestones: c.milestones.map((m) => ({ id: m.id, day: m.day, detail: m.detail })),
    goalProgress: state.goal.spec.goal.kind === "none" ? null : goalProgress(state, c),
    interviews: c.interviews.map((i) => ({
      id: i.id,
      day: i.day,
      productId: i.productId,
      productName: i.productName,
      outlet: i.outlet,
      mean: i.mean,
      answers: i.answers.map((a) => ({ ...a })),
    })),
    nextInterviewDay:
      c.lastInterviewDay === null
        ? null
        : c.lastInterviewDay + getBalance(state.balance).economy.interviews.cooldownDays,
    discoveries: c.discoveries.map((d) => ({
      id: d.id,
      versionId: d.versionId,
      versionLabel: c.versions[d.versionId]?.label ?? "",
      day: d.day,
      kind: d.kind,
      about: d.about,
      title: d.title,
      category: d.category,
      tag: d.tag,
      value: d.value,
      compared: d.compared,
      evidence: d.evidence ? { prompt: d.evidence.prompt, reply: d.evidence.reply } : null,
      published: d.published,
    })),
    fund: { balance: c.fund.balance },
    showcases: c.showcases.map((x) => ({
      id: x.id,
      day: x.day,
      game: x.game,
      difficulty: x.difficulty,
      productId: x.productId,
      versionId: x.versionId,
      productName: x.productName,
      status: x.status,
      score: x.score,
      maxScore: x.maxScore,
    })),
    shop: shopView(getBalance(state.balance)),
    reports: c.reports.slice(-MONTHS_SHOWN).map((r) => ({
      ...r,
      growthByGroup: pick(GROUP_IDS, r.growthByGroup),
      best: r.best ? { ...r.best, author: { ...r.best.author } } : null,
      worst: r.worst ? { ...r.worst, author: { ...r.worst.author } } : null,
    })),
    dataCenters: Object.values(c.dataCenters)
      .sort((a, b) => a.startDay - b.startDay || cmp(a.id, b.id))
      .map((d) => ({
        id: d.id,
        region: d.region,
        startDay: d.startDay,
        openDay: d.openDay,
        open: d.openDay <= state.day,
      })),
    campaigns: Object.values(c.campaigns)
      .sort((a, b) => a.startDay - b.startDay || cmp(a.id, b.id))
      .map((k) => ({
        id: k.id,
        region: k.region,
        group: k.group,
        budgetPerDay: k.budgetPerDay,
        startDay: k.startDay,
        endDay: k.endDay,
        spent: k.spent,
      })),
    history: Object.fromEntries(
      c.history.map(({ day, ...sample }) => [
        String(day),
        { ...sample, customers: { ...sample.customers }, regions: { ...sample.regions } },
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
  const byRegion = {} as PrivateCompany["customers"]["byRegion"];
  for (const region of REGION_IDS) {
    const groups = {} as Record<GroupId, number>;
    for (const group of GROUP_IDS) {
      let n = 0;
      for (const plan of PLAN_IDS) n += customersOf(state.market, c.id, region, group, plan);
      groups[group] = Math.round(n);
    }
    byRegion[region] = groups;
  }
  return { total: Math.round(totalCustomers(state.market, c.id)), byGroup, byRegion };
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
    income: pick(INCOME_CATEGORIES, m.income),
    costs: pick(COST_CATEGORIES, m.costs),
    revenueByGroup: pick(GROUP_IDS, m.revenueByGroup),
    revenueByPlan: pick(PLAN_IDS, m.revenueByPlan),
    revenueByRegion: pick(REGION_IDS, m.revenueByRegion),
    customersEnd: m.customersEnd,
    customersEndByGroup: pick(GROUP_IDS, m.customersEndByGroup),
    financing: {
      loans: m.financing.loans,
      investors: m.financing.investors,
      repaid: m.financing.repaid,
    },
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
    ...(i.stars ? { stars: i.stars } : {}),
    ...(i.press ? { press: true as const } : {}),
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
    // Absent from plans logged before sim-5.
    tutoring: p.tutoring ? { ...p.tutoring } : null,
    learnFromCustomers: p.learnFromCustomers ?? false,
    coaching: p.coaching ?? false,
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

/** What the playtest pack's purchases cost, so the client never hard-codes a price. */
function shopView(b: Balance): ShopView {
  const e = b.economy;
  return {
    showcase: { fee: e.showcases.fee, cooldownDays: e.showcases.cooldownDays },
    sponsorships: SPONSOR_KINDS.map((kind) => ({
      kind,
      price: e.sponsorships.kinds[kind].price,
      days: e.sponsorships.kinds[kind].days,
      groups: [...e.sponsorships.kinds[kind].groups],
    })),
    looks: Object.entries(e.looks).map(([id, price]) => ({ id, price })),
    apps: APP_KINDS.map((kind) => ({
      kind,
      skill: e.apps.kinds[kind].skill,
      group: e.apps.kinds[kind].group,
      setup: e.apps.kinds[kind].setup,
      upkeep: e.apps.kinds[kind].upkeep,
    })),
  };
}
