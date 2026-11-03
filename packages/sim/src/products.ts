import type { Command, PlanId } from "@airace/protocol";
import { type Balance, getBalance } from "./balance.ts";
import { tellNews } from "./feed.ts";
import { book } from "./ledger.ts";
import type {
  CompanyState,
  FameState,
  PlansState,
  ProductState,
  SystemEvent,
  WorldState,
} from "./state.ts";
import { formatMoney } from "./training.ts";

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

type ProductCommand = Extract<
  Command,
  { type: "launchProduct" | "setPlan" | "setPrice" | "retireProduct" }
>;

/** A new company's business: no products, every plan closed at the default prices. */
export function newCompanyBusiness(b: Balance): {
  products: Record<string, ProductState>;
  plans: PlansState;
  fame: FameState;
  announced: CompanyState["announced"];
} {
  const { prices, fame } = b.economy;
  return {
    products: {},
    plans: {
      free: { productId: null },
      personal: { productId: null, price: prices.personal.default },
      business: { productId: null, price: prices.business.default },
    },
    fame: { reputation: fame.reputationStart, followers: 0, newsSentiment: 0 },
    announced: { customers: 0, firstPaying: false },
  };
}

/** The product with this id in any company. Product ids are unique across a world. */
export function findProduct(
  s: WorldState,
  productId: string,
): { company: CompanyState; product: ProductState } | null {
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (company && Object.hasOwn(company.products, productId)) {
      const product = company.products[productId];
      if (product) return { company, product };
    }
  }
  return null;
}

/** The live product behind a plan, or null when the plan is closed. */
export function liveProductFor(company: CompanyState, plan: PlanId): ProductState | null {
  const id = company.plans[plan].productId;
  const product = id ? company.products[id] : undefined;
  return product?.status === "live" ? product : null;
}

/** Products in countdown or live, which count against the lineup limit. */
function onSale(company: CompanyState): ProductState[] {
  return Object.values(company.products).filter((p) => p.status !== "retired");
}

export function productCommand(
  s: WorldState,
  company: CompanyState,
  cmd: ProductCommand,
  newId: string | undefined,
): Handled {
  const economy = getBalance(s.balance).economy;
  switch (cmd.type) {
    case "launchProduct": {
      if (s.phase !== "running") return no("The world is not running yet.");
      if (!newId) return no("Missing product id.");
      if (findProduct(s, newId)) return no("That product id is already used.");
      const version = company.versions[cmd.versionId];
      if (!version) return no("That AI isn't one of yours.");
      if (version.scores === null) {
        return no("Wait until your AI has been scored before launching it.");
      }
      const { min, max } = economy.countdown;
      if (cmd.countdownDays < min || cmd.countdownDays > max) {
        return no(`The countdown must be ${min}–${max} game days.`);
      }
      if (onSale(company).length >= economy.maxProducts) {
        return no(`You can sell at most ${economy.maxProducts} AIs at once. Retire one first.`);
      }
      const lower = cmd.name.toLowerCase();
      if (onSale(company).some((p) => p.name.toLowerCase() === lower)) {
        return no("You already sell an AI with that name.");
      }
      const keynote = cmd.keynote ?? null;
      if (keynote && company.cash < economy.keynote.fee) {
        return no(
          `A launch keynote costs ${formatMoney(economy.keynote.fee)}. You have ${formatMoney(company.cash)}.`,
        );
      }
      if (keynote) book(company, "events", economy.keynote.fee);
      company.products[newId] = {
        id: newId,
        name: cmd.name,
        versionId: cmd.versionId,
        status: "countdown",
        announcedDay: s.day,
        launchDay: s.day + cmd.countdownDays,
        hype: 0,
        launchPlans: [...new Set(cmd.plans)],
        feedbackKeys: [],
        keynote: keynote
          ? { questionId: keynote.questionId, status: "waiting", result: null, applied: false }
          : null,
      };
      return OK;
    }
    case "setPlan": {
      if (cmd.productId !== null) {
        const product = company.products[cmd.productId];
        if (!product || product.status === "retired") return no("That AI isn't one of yours.");
        if (product.status !== "live")
          return no("That AI isn't on sale yet. Wait for its launch day.");
      }
      company.plans[cmd.plan].productId = cmd.productId;
      return OK;
    }
    case "setPrice": {
      const { min, max } = economy.prices[cmd.plan];
      if (cmd.price < min || cmd.price > max) {
        const label = cmd.plan === "personal" ? "Personal prices" : "Business prices";
        return no(`${label} are ${formatMoney(min)}–${formatMoney(max)} a month.`);
      }
      company.plans[cmd.plan].price = cmd.price;
      return OK;
    }
    case "retireProduct": {
      const product = company.products[cmd.productId];
      if (!product || product.status === "retired") return no("That AI isn't one of yours.");
      const usedBy = (["free", "personal", "business"] as const).some(
        (plan) => company.plans[plan].productId === product.id,
      );
      if (usedBy) return no("Move its plans to another AI first.");
      product.status = "retired";
      return OK;
    }
  }
}

/** The daily product step: countdowns build hype, and launch days put products on sale. */
export function productsDay(s: WorldState): void {
  const { hypePerDay } = getBalance(s.balance).economy;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (!company) continue;
    const { reputation, followers } = company.fame;
    for (const productId of Object.keys(company.products).sort()) {
      const product = company.products[productId];
      if (product?.status !== "countdown") continue;
      if (s.day >= product.launchDay) {
        product.status = "live";
        for (const plan of product.launchPlans) company.plans[plan].productId = product.id;
        // A keynote answered before launch day: today's launch news quotes it.
        if (product.keynote?.status === "done") tellKeynote(s, company, product);
      } else {
        product.hype += hypePerDay * (reputation / 100 + followers / (followers + 1000));
      }
    }
  }
}

/** The keynote's story, once: the real reply on stage, or a flop (playtest pack §5). */
function tellKeynote(s: WorldState, company: CompanyState, product: ProductState): void {
  const k = product.keynote;
  if (!k?.result || k.applied) return;
  k.applied = true;
  const { score, reply } = k.result;
  const stumble = getBalance(s.balance).economy.keynote.stumble;
  const headline =
    score < stumble
      ? `${company.name}'s keynote flops: on stage, ${product.name} said "${reply}"`
      : `Live on stage, ${product.name} said: "${reply}"`;
  tellNews(s, company, "keynote", headline, 2 * score - 1);
}

/** The keynote's live demo came back: kept for launch day, or told at once if the AI is out. */
export function applyKeynote(
  s: WorldState,
  e: Extract<SystemEvent, { type: "keynoteResolved" }>,
): Handled {
  const found = findProduct(s, e.productId);
  const k = found?.product.keynote;
  if (!found || k?.status !== "waiting") return no("This keynote is over.");
  const { company, product } = found;
  if ("failed" in e && e.failed) {
    k.status = "failed";
    k.applied = true;
    book(company, "events", -getBalance(s.balance).economy.keynote.fee);
    return OK;
  }
  const r = e as { prompt?: unknown; reply?: unknown; score?: unknown };
  if (
    typeof r.prompt !== "string" ||
    typeof r.reply !== "string" ||
    typeof r.score !== "number" ||
    !(r.score >= 0 && r.score <= 1)
  ) {
    return no("That keynote result isn't valid.");
  }
  k.status = "done";
  k.result = { prompt: r.prompt, reply: r.reply, score: r.score };
  if (product.status === "live") tellKeynote(s, company, product);
  return OK;
}
