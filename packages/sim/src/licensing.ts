import {
  type Command,
  GROUP_IDS,
  type GroupId,
  LICENSE_PRICE_CENTS,
  PLAN_IDS,
  REGION_IDS,
} from "@airace/protocol";
import { tellNews } from "./feed.ts";
import { book } from "./ledger.ts";
import { customersOf } from "./market.ts";
import type {
  CompanyState,
  Deal,
  LicenseOffer,
  LicenseProposal,
  ProductState,
  VersionState,
  WorldState,
} from "./state.ts";

// Licensing (C44): a company rents one of its AIs to another company, which uses it for
// chosen customer groups of one of its products, for royalties per customer a month.
// Offers are public; proposals are private between the two companies; deals run for a
// term of 30-day months.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

/** Game days in a license month. */
export const LICENSE_MONTH_DAYS = 30;
/** A proposal nobody answers lapses after this many days. */
export const PROPOSAL_DAYS = 30;

const ON_SALE = "Pick one of your AIs on sale.";
const OWN = "You can't license your own AI.";
const PRICE = `Prices are $0.10–$20 a customer a month.`;
const OVERLAP = "That AI already uses a licensed AI for one of those groups.";
const EXCLUSIVE = "That AI is licensed exclusively.";

type LicenseCommand = Extract<
  Command,
  {
    type:
      | "offerLicense"
      | "withdrawOffer"
      | "takeLicense"
      | "proposeLicense"
      | "answerProposal"
      | "endLicense";
  }
>;

/** The running license that powers `productId` for `group`, if any. */
export function activeLicense(s: WorldState, productId: string, group: GroupId): Deal | null {
  return (
    s.licenses.deals.find((d) => d.productId === productId && d.groups.includes(group)) ?? null
  );
}

/** The AI that answers `productId`'s `group`: a licensed one, or null for the product's own. */
export function licensedVersion(
  s: WorldState,
  productId: string,
  group: GroupId,
): VersionState | null {
  const deal = activeLicense(s, productId, group);
  if (!deal) return null;
  const licensor = s.companies[deal.licensorId];
  return licensor && Object.hasOwn(licensor.versions, deal.versionId)
    ? (licensor.versions[deal.versionId] ?? null)
    : null;
}

/** The groups in news copy. */
const GROUP_WORDS: Record<GroupId, string> = {
  students: "students",
  creators: "creators",
  developers: "developers",
  businesses: "businesses",
  everyday: "everyday users",
  researchers: "researchers",
};

/** "students", "students and creators", "students, creators and developers". */
export function groupList(groups: readonly GroupId[]): string {
  const words = inOrder(groups).map((g) => GROUP_WORDS[g]);
  if (words.length <= 1) return words.join("");
  return `${words.slice(0, -1).join(", ")} and ${words.at(-1)}`;
}

/** The company's newest live product (latest launch, then id), if any. */
export function newestLive(company: CompanyState): ProductState | null {
  let best: ProductState | null = null;
  for (const p of Object.values(company.products)) {
    if (p.status !== "live") continue;
    if (!best || p.launchDay > best.launchDay || (p.launchDay === best.launchDay && p.id > best.id))
      best = p;
  }
  return best;
}

/** The company's newest live product built on `versionId`: the AI is on sale. */
export function onSaleAs(company: CompanyState, versionId: string): ProductState | null {
  let best: ProductState | null = null;
  for (const p of Object.values(company.products)) {
    if (p.status !== "live" || p.versionId !== versionId) continue;
    if (!best || p.launchDay > best.launchDay || (p.launchDay === best.launchDay && p.id > best.id))
      best = p;
  }
  return best;
}

/** The six groups in their usual order, each once. */
function inOrder(groups: readonly GroupId[]): GroupId[] {
  return GROUP_IDS.filter((g) => groups.includes(g));
}

function nextId(s: WorldState, kind: "offer" | "proposal" | "deal"): string {
  s.licenses.seq += 1;
  return `${kind}-${s.licenses.seq}`;
}

const priceOk = (cents: number) =>
  Number.isInteger(cents) && cents >= LICENSE_PRICE_CENTS.min && cents <= LICENSE_PRICE_CENTS.max;

const exclusivelyLicensed = (s: WorldState, versionId: string) =>
  s.licenses.deals.some((d) => d.versionId === versionId && d.exclusive);

/** Why `licensee`'s product can't take `versionId` for `groups`, or null when it can. */
function takeProblem(
  s: WorldState,
  licensor: CompanyState,
  licensee: CompanyState,
  versionId: string,
  productId: string,
  groups: readonly GroupId[],
): string | null {
  if (licensor.id === licensee.id) return OWN;
  const product = Object.hasOwn(licensee.products, productId)
    ? licensee.products[productId]
    : undefined;
  if (product?.status !== "live") return ON_SALE;
  if (!onSaleAs(licensor, versionId)) return "That AI isn't on sale any more.";
  if (groups.some((g) => activeLicense(s, productId, g))) return OVERLAP;
  return null;
}

function startDeal(
  s: WorldState,
  terms: Omit<Deal, "id" | "startDay" | "endDay" | "owed"> & { months: number },
): Deal {
  const { months, ...rest } = terms;
  const deal: Deal = {
    id: nextId(s, "deal"),
    ...rest,
    groups: inOrder(rest.groups),
    startDay: s.day,
    endDay: s.day + months * LICENSE_MONTH_DAYS,
    owed: 0,
  };
  s.licenses.deals.push(deal);
  const licensee = s.companies[deal.licenseeId];
  const licensor = s.companies[deal.licensorId];
  if (licensee && licensor) {
    tellNews(
      s,
      licensee,
      "license",
      `${licensee.name} licenses ${licensor.name}'s ${deal.aiName} for its ${groupList(deal.groups)}`,
      0.3,
    );
  }
  return deal;
}

/** The licensee's customers today in the deal's groups, on every plan its product powers. */
export function licensedCustomers(s: WorldState, deal: Deal): number {
  const licensee = s.companies[deal.licenseeId];
  if (!licensee) return 0;
  let n = 0;
  for (const plan of PLAN_IDS) {
    if (licensee.plans[plan].productId !== deal.productId) continue;
    for (const region of REGION_IDS) {
      for (const group of deal.groups) n += customersOf(s.market, licensee.id, region, group, plan);
    }
  }
  return n;
}

/** Pays what the deal has built up, from licensee to licensor (whole dollars, both sides alike). */
function settle(s: WorldState, deal: Deal, extra = 0): void {
  const dollars = Math.round(deal.owed + extra);
  deal.owed = 0;
  const licensee = s.companies[deal.licenseeId];
  const licensor = s.companies[deal.licensorId];
  if (!licensee || !licensor || dollars <= 0) return;
  book(licensee, "licenses", dollars);
  book(licensor, "licenseIncome", dollars);
}

/** Ends a deal: pays what it built up (and `extra`), and tells the news. */
function endDeal(s: WorldState, deal: Deal, extra = 0): void {
  settle(s, deal, extra);
  s.licenses.deals = s.licenses.deals.filter((d) => d.id !== deal.id);
  const licensee = s.companies[deal.licenseeId];
  if (licensee) {
    tellNews(s, licensee, "license", `${licensee.name}'s license of ${deal.aiName} ends`, 0);
  }
}

/** At a month's close, every deal pays what it built up that month. Call before the books close. */
export function settleLicenses(s: WorldState): void {
  for (const deal of s.licenses.deals) settle(s, deal);
}

/**
 * The daily licensing step, after products: deals whose term is over, whose AI is no longer on
 * sale or whose product closed end (paying what they built up); the rest build up a 30th of
 * their monthly royalty for today's customers. Offers of AIs no longer on sale close, and
 * proposals lapse.
 */
export function licensesDay(s: WorldState): void {
  const L = s.licenses;
  for (const deal of [...L.deals]) {
    const licensor = s.companies[deal.licensorId];
    const licensee = s.companies[deal.licenseeId];
    // A rival moves its license to the AI that took over its plans.
    if (licensee?.rival && licensee.products[deal.productId]?.status !== "live") {
      const next = newestLive(licensee);
      if (next && !deal.groups.some((g) => activeLicense(s, next.id, g))) deal.productId = next.id;
    }
    const over =
      s.day >= deal.endDay ||
      !licensor ||
      !licensee ||
      !onSaleAs(licensor, deal.versionId) ||
      licensee.products[deal.productId]?.status !== "live";
    if (over) endDeal(s, deal);
  }
  for (const deal of L.deals) {
    deal.owed += ((deal.priceCents / 100) * licensedCustomers(s, deal)) / LICENSE_MONTH_DAYS;
  }
  L.offers = L.offers.filter((o) => {
    const licensor = s.companies[o.licensorId];
    return licensor !== undefined && onSaleAs(licensor, o.versionId) !== null;
  });
  L.proposals = L.proposals.filter((p) => {
    const licensor = s.companies[p.licensorId];
    return (
      s.day < p.day + PROPOSAL_DAYS &&
      licensor !== undefined &&
      onSaleAs(licensor, p.versionId) !== null &&
      s.companies[p.licenseeId]?.products[p.productId]?.status === "live"
    );
  });
}

/** Posts an offer of `licensor`'s AI. Rivals use it too. */
export function makeOffer(
  s: WorldState,
  licensor: CompanyState,
  t: {
    versionId: string;
    priceCents: number;
    months: number;
    groups: GroupId[];
    exclusive: boolean;
  },
): Handled {
  if (!priceOk(t.priceCents)) return no(PRICE);
  const product = Object.hasOwn(licensor.versions, t.versionId)
    ? onSaleAs(licensor, t.versionId)
    : null;
  if (!product) return no(ON_SALE);
  if (s.licenses.offers.some((o) => o.versionId === t.versionId)) {
    return no("You already offer that AI. Withdraw that offer first.");
  }
  if (exclusivelyLicensed(s, t.versionId)) return no(EXCLUSIVE);
  if (t.exclusive && s.licenses.deals.some((d) => d.versionId === t.versionId)) {
    return no("An exclusive offer needs an AI nobody licenses yet.");
  }
  const offer: LicenseOffer = {
    id: nextId(s, "offer"),
    licensorId: licensor.id,
    versionId: t.versionId,
    aiName: product.name,
    priceCents: t.priceCents,
    months: t.months,
    groups: inOrder(t.groups),
    exclusive: t.exclusive,
    day: s.day,
  };
  s.licenses.offers.push(offer);
  return OK;
}

/** `licensee` takes `offerId` for its product `productId` and `groups`. Rivals use it too. */
export function takeOffer(
  s: WorldState,
  licensee: CompanyState,
  offerId: string,
  productId: string,
  groups: readonly GroupId[],
): Handled {
  const offer = s.licenses.offers.find((o) => o.id === offerId);
  const licensor = offer ? s.companies[offer.licensorId] : undefined;
  if (!offer || !licensor) return no("That offer has been taken.");
  if (!groups.every((g) => offer.groups.includes(g))) {
    return no("That offer doesn't cover those groups.");
  }
  const problem = takeProblem(s, licensor, licensee, offer.versionId, productId, groups);
  if (problem) return no(problem);
  // An exclusive offer is for an AI nobody else licenses (a proposal may have come first).
  if (offer.exclusive && s.licenses.deals.some((d) => d.versionId === offer.versionId)) {
    return no("That offer has been taken.");
  }
  startDeal(s, {
    licensorId: licensor.id,
    licenseeId: licensee.id,
    versionId: offer.versionId,
    aiName: offer.aiName,
    productId,
    priceCents: offer.priceCents,
    groups: [...groups],
    exclusive: offer.exclusive,
    months: offer.months,
  });
  if (offer.exclusive) s.licenses.offers = s.licenses.offers.filter((o) => o.id !== offer.id);
  return OK;
}

/** The company that made `versionId`, if any. */
function makerOf(s: WorldState, versionId: string): CompanyState | null {
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (company && Object.hasOwn(company.versions, versionId)) return company;
  }
  return null;
}

/** `licensee` asks the maker of `versionId` for a license. Rivals use it too. */
export function makeProposal(
  s: WorldState,
  licensee: CompanyState,
  t: {
    versionId: string;
    productId: string;
    priceCents: number;
    months: number;
    groups: GroupId[];
  },
): Handled {
  if (!priceOk(t.priceCents)) return no(PRICE);
  const licensor = makerOf(s, t.versionId);
  if (!licensor) return no("That AI isn't on sale.");
  if (licensor.id === licensee.id) return no(OWN);
  if (licensor.rival) return no("Rivals license their AIs only through their offers.");
  const product = onSaleAs(licensor, t.versionId);
  if (!product) return no("That AI isn't on sale.");
  const problem = takeProblem(s, licensor, licensee, t.versionId, t.productId, t.groups);
  if (problem) return no(problem);
  if (exclusivelyLicensed(s, t.versionId)) return no(EXCLUSIVE);
  if (
    s.licenses.proposals.some((p) => p.licenseeId === licensee.id && p.versionId === t.versionId)
  ) {
    return no("You've already asked about that AI.");
  }
  const proposal: LicenseProposal = {
    id: nextId(s, "proposal"),
    licensorId: licensor.id,
    licenseeId: licensee.id,
    versionId: t.versionId,
    aiName: product.name,
    productId: t.productId,
    priceCents: t.priceCents,
    months: t.months,
    groups: inOrder(t.groups),
    day: s.day,
  };
  s.licenses.proposals.push(proposal);
  return OK;
}

/** The maker answers a proposal: accepting starts the deal. */
function answerProposal(
  s: WorldState,
  licensor: CompanyState,
  proposalId: string,
  accept: boolean,
): Handled {
  const proposal = s.licenses.proposals.find((p) => p.id === proposalId);
  if (!proposal) return no("That request has lapsed.");
  if (proposal.licensorId !== licensor.id) return no("That request isn't yours to answer.");
  const licensee = s.companies[proposal.licenseeId];
  if (accept) {
    if (!licensee) return no("That request has lapsed.");
    const problem = takeProblem(
      s,
      licensor,
      licensee,
      proposal.versionId,
      proposal.productId,
      proposal.groups,
    );
    if (problem) return no(problem === ON_SALE ? "Their AI is no longer on sale." : problem);
    if (exclusivelyLicensed(s, proposal.versionId)) return no(EXCLUSIVE);
    startDeal(s, {
      licensorId: licensor.id,
      licenseeId: licensee.id,
      versionId: proposal.versionId,
      aiName: proposal.aiName,
      productId: proposal.productId,
      priceCents: proposal.priceCents,
      groups: proposal.groups,
      exclusive: false,
      months: proposal.months,
    });
  }
  s.licenses.proposals = s.licenses.proposals.filter((p) => p.id !== proposal.id);
  return OK;
}

export function licenseCommand(s: WorldState, company: CompanyState, cmd: LicenseCommand): Handled {
  if (s.phase !== "running") return no("The world is not running yet.");
  switch (cmd.type) {
    case "offerLicense":
      return makeOffer(s, company, cmd);
    case "withdrawOffer": {
      const offer = s.licenses.offers.find((o) => o.id === cmd.offerId);
      if (!offer) return no("That offer has been taken.");
      if (offer.licensorId !== company.id) return no("That offer isn't yours.");
      s.licenses.offers = s.licenses.offers.filter((o) => o.id !== offer.id);
      return OK;
    }
    case "takeLicense":
      return takeOffer(s, company, cmd.offerId, cmd.productId, cmd.groups);
    case "proposeLicense":
      return makeProposal(s, company, cmd);
    case "answerProposal":
      return answerProposal(s, company, cmd.proposalId, cmd.accept);
    case "endLicense": {
      const deal = s.licenses.deals.find((d) => d.id === cmd.dealId);
      if (!deal || (deal.licenseeId !== company.id && deal.licensorId !== company.id)) {
        return no("That license has ended.");
      }
      if (deal.licenseeId !== company.id) {
        return no("Only the company that licenses it can end it early.");
      }
      // Ending early costs one more month's royalties at today's customers.
      endDeal(s, deal, (deal.priceCents / 100) * licensedCustomers(s, deal));
      return OK;
    }
  }
}
