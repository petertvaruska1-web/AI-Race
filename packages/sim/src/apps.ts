import { APP_NAMES, type AppKind, type Command } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { postAppReview } from "./feed.ts";
import { book } from "./ledger.ts";
import type { AppState, CompanyState, SystemEvent, WorldState } from "./state.ts";
import { formatMoney } from "./training.ts";

// Partner apps (playtest pack §5): small apps built on one of the company's products,
// each betting on one skill. Users come as the AI's *measured* score in that skill and
// the app's star rating allow; real replies to fresh questions become its reviews.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

type AppCommand = Extract<Command, { type: "fundApp" | "closeApp" }>;

export function appCommand(
  s: WorldState,
  company: CompanyState,
  cmd: AppCommand,
  newId: string | undefined,
): Handled {
  if (cmd.type === "closeApp") {
    const app = company.apps[cmd.appId];
    if (app?.status !== "live") return no("That app isn't running.");
    app.status = "closed";
    app.users = 0;
    return OK;
  }
  if (!newId) return no("Missing app id.");
  const kind = getBalance(s.balance).economy.apps.kinds[cmd.kind];
  if (company.products[cmd.productId]?.status !== "live") {
    return no("Build apps on an AI that is on sale.");
  }
  if (Object.values(company.apps).some((a) => a.kind === cmd.kind && a.status === "live")) {
    return no(`You already have a ${APP_NAMES[cmd.kind]} app.`);
  }
  if (company.cash < kind.setup) {
    return no(
      `Setting it up costs ${formatMoney(kind.setup)}. You have ${formatMoney(company.cash)}.`,
    );
  }
  if (company.apps[newId]) return no("That app id is already used.");
  book(company, "appCosts", kind.setup);
  company.apps[newId] = {
    id: newId,
    kind: cmd.kind,
    productId: cmd.productId,
    startDay: s.day,
    users: 0,
    rating: 3,
    status: "live",
    reviews: [],
    reviewKeys: [],
  };
  return OK;
}

/** How well the app's AI fits its job, 0-1, from the version's measured score in its skill. */
export function appFit(s: WorldState, company: CompanyState, app: AppState): number {
  const e = getBalance(s.balance).economy.apps;
  const product = company.products[app.productId];
  const version = product ? company.versions[product.versionId] : undefined;
  const score = version?.scores?.[e.kinds[app.kind].skill];
  if (typeof score !== "number") return e.unmeasuredFit;
  return Math.max(0, Math.min(1, score / 100)) ** e.fitPower;
}

/** A day of every running app: users move toward what the AI and the rating allow; money moves. */
export function appsDay(s: WorldState): void {
  const e = getBalance(s.balance).economy.apps;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id] as CompanyState;
    for (const app of Object.values(company.apps)) {
      if (app.status !== "live") continue;
      const kind = e.kinds[app.kind];
      book(company, "appCosts", kind.upkeep / 30);
      if (company.products[app.productId]?.status !== "live") {
        // Its AI left sale: people drift away until it is closed or moved.
        app.users *= 1 - e.dailyGrowth;
        continue;
      }
      const target = kind.capacity * appFit(s, company, app) * (app.rating / 3);
      app.users += (target - app.users) * e.dailyGrowth;
      book(company, "appRevenue", (app.users * kind.revenuePerUser) / 30);
    }
  }
}

/** Applies a batch of app-store reviews once: stars, the rating, and a post quoting a real reply. */
export function applyAppReviews(
  s: WorldState,
  e: Extract<SystemEvent, { type: "appReviewed" }>,
): Handled {
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id] as CompanyState;
    const app = company.apps[e.appId];
    if (!app) continue;
    if (app.reviewKeys.includes(e.key)) return no("Those reviews are already counted.");
    const valid =
      Array.isArray(e.reviews) &&
      e.reviews.length > 0 &&
      e.reviews.every(
        (r) =>
          typeof r?.prompt === "string" &&
          typeof r.reply === "string" &&
          typeof r.score === "number" &&
          r.score >= 0 &&
          r.score <= 1,
      );
    if (!valid) return no("Those reviews aren't valid.");
    const kept = getBalance(s.balance).economy.apps.reviewsKept;
    for (const r of e.reviews) {
      const stars = Math.round(1 + 4 * r.score);
      app.reviews.push({ day: s.day, prompt: r.prompt, reply: r.reply, score: r.score, stars });
    }
    if (app.reviews.length > kept) app.reviews.splice(0, app.reviews.length - kept);
    app.rating = app.reviews.reduce((sum, r) => sum + r.stars, 0) / app.reviews.length;
    app.reviewKeys.push(e.key);
    if (app.reviewKeys.length > 20) app.reviewKeys.splice(0, app.reviewKeys.length - 20);
    // The review furthest from "fine" makes the post (real reply, unchanged: R1).
    const loud = e.reviews.reduce((a, b) =>
      Math.abs(b.score - 0.5) > Math.abs(a.score - 0.5) ? b : a,
    );
    const group = getBalance(s.balance).economy.apps.kinds[app.kind as AppKind].group;
    postAppReview(s, company, APP_NAMES[app.kind], group, loud.prompt, loud.reply, loud.score);
    return OK;
  }
  return no("There's no such app.");
}
