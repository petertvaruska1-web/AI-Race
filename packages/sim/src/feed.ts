import {
  BENCH_CATEGORIES,
  type BenchCategory,
  GROUP_IDS,
  type GroupId,
  type NewsTopic,
  type PostTone,
  REGION_IDS,
} from "@airace/protocol";
import { type Economy, getBalance } from "./balance.ts";
import { noteNews } from "./fame.ts";
import { customersOf, totalCustomers } from "./market.ts";
import { findProduct } from "./products.ts";
import { rngInt } from "./rng.ts";
import { drawAuthor } from "./spotlights.ts";
import type {
  CompanyState,
  FeedbackResult,
  FeedItem,
  NewsItem,
  PostItem,
  ProductState,
  SystemEvent,
  WorldState,
} from "./state.ts";

// The feed (game-layer design §7): customer posts that quote real replies
// verbatim inside a game-written frame (R1), and news stories the game writes
// about real events. Headlines are game commentary, never an AI's words.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

/** Feedback batches a product remembers, so a result is never applied twice. */
const KEYS_KEPT = 20;
/** Customer counts that make the news, once each. */
const CUSTOMER_MARKS = [1_000, 10_000, 100_000, 1_000_000];
/** The bench categories in plain words (R7). */
const SKILL_WORDS: Record<BenchCategory, string> = {
  language: "language",
  reasoning: "reasoning",
  pattern: "patterns",
  knowledge: "knowledge",
  coding: "coding",
  creativity: "creativity",
  consistency: "consistency",
  instruction: "following instructions",
};
const TONE_WORDS: Record<PostTone, string> = {
  praise: "praise for",
  joke: "joke about",
  roast: "roast of",
};

const grouped = (n: number) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");

type NewFeedItem = Omit<PostItem, "id"> | Omit<NewsItem, "id">;

/** Appends an item with the next id, dropping the oldest past the cap. */
function addItem(s: WorldState, item: NewFeedItem): void {
  s.feedSeq += 1;
  s.feed.push({ ...item, id: `f${s.feedSeq}` } as FeedItem);
  const cap = getBalance(s.balance).economy.feedCap;
  if (s.feed.length > cap) s.feed.splice(0, s.feed.length - cap);
}

/** Adds a news story about a company, and lets it move the company's news mood. */
export function tellNews(
  s: WorldState,
  company: CompanyState,
  topic: NewsTopic,
  headline: string,
  sentiment: number,
): void {
  addItem(s, { kind: "news", day: s.day, companyId: company.id, sentiment, topic, headline });
  noteNews(getBalance(s.balance).economy, company, sentiment);
}

function toneOf(score: number): PostTone {
  if (score >= 0.75) return "praise";
  if (score <= 0.25) return "roast";
  return "joke";
}

function validResult(r: unknown): r is FeedbackResult {
  if (typeof r !== "object" || r === null) return false;
  const x = r as Record<string, unknown>;
  return (
    (GROUP_IDS as readonly unknown[]).includes(x.group) &&
    typeof x.questionId === "string" &&
    typeof x.question === "string" &&
    typeof x.reply === "string" &&
    typeof x.score === "number" &&
    x.score >= 0 &&
    x.score <= 1
  );
}

/** The group's spotlight customer half the time (world rng), otherwise a drawn one. */
function postAuthor(s: WorldState, group: GroupId): PostItem["author"] {
  const coin = rngInt(s.rng, 2);
  s.rng = coin.next;
  const spot = s.spotlights.find((p) => p.group === group);
  if (spot && coin.value === 0) {
    return { name: spot.name, handle: spot.handle, group, spotlightId: spot.id };
  }
  const drawn = drawAuthor(s.rng);
  s.rng = drawn.rng;
  return { name: drawn.name, handle: drawn.handle, group, spotlightId: null };
}

/** Each group's satisfaction with the product moves toward the batch's mean score. */
function updateSatisfaction(
  s: WorldState,
  economy: Economy,
  productId: string,
  results: FeedbackResult[],
): void {
  const { satisfactionAlpha, satisfactionStart } = economy.feedback;
  const byGroup = s.market.satisfaction[productId] ?? {};
  for (const group of GROUP_IDS) {
    const scores = results.filter((r) => r.group === group).map((r) => r.score);
    if (scores.length === 0) continue;
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const old = byGroup[group] ?? satisfactionStart;
    byGroup[group] = old + satisfactionAlpha * (mean - old);
  }
  s.market.satisfaction[productId] = byGroup;
}

/**
 * The best and the worst reply become posts. With enough followers, the more
 * extreme of them goes viral (a roast wins a tie: bad news travels faster).
 * A rival's batch makes one post, its most extreme reply, and never goes viral,
 * so four rivals don't flood the feed.
 */
function postBestAndWorst(
  s: WorldState,
  economy: Economy,
  company: CompanyState,
  product: ProductState,
  results: FeedbackResult[],
): void {
  const first = results[0];
  if (!first) return;
  let best = first;
  let worst = first;
  for (const r of results) {
    if (r.score > best.score) best = r;
    if (r.score < worst.score) worst = r;
  }
  const extremity = (score: number) => Math.abs(score - 0.5);
  const rival = company.rival !== null;
  let chosen = best === worst ? [best] : [best, worst];
  if (rival) {
    chosen = [results.reduce((a, b) => (extremity(b.score) > extremity(a.score) ? b : a), first)];
  }
  const made: { post: Omit<PostItem, "id">; score: number }[] = [];
  for (const r of chosen) {
    const post: Omit<PostItem, "id"> = {
      kind: "post",
      day: s.day,
      companyId: company.id,
      sentiment: 2 * r.score - 1,
      author: postAuthor(s, r.group),
      productName: product.name,
      question: r.question,
      reply: r.reply,
      tone: toneOf(r.score),
    };
    addItem(s, post);
    made.push({ post, score: r.score });
  }
  const { viralScore, viralFollowers } = economy.fame;
  if (rival || company.fame.followers < viralFollowers) return;
  const viral = made
    .filter((m) => m.score >= viralScore || 1 - m.score >= viralScore)
    .sort((a, b) => extremity(b.score) - extremity(a.score) || a.score - b.score)[0]?.post;
  if (!viral) return;
  company.fame.followers += viralFollowers / 10;
  const headline = `${viral.author.handle}'s ${TONE_WORDS[viral.tone]} ${product.name} is going viral`;
  tellNews(s, company, "viral", headline, viral.sentiment);
}

/** Applies a scored feedback batch once: satisfaction, then posts (and maybe a viral story). */
export function applyFeedback(
  s: WorldState,
  e: Extract<SystemEvent, { type: "feedbackScored" }>,
): Handled {
  const found = findProduct(s, e.productId);
  if (!found) return no("Unknown product.");
  const { company, product } = found;
  if (product.status !== "live") return no("This product is not on sale.");
  if (product.feedbackKeys.includes(e.key)) return no("This feedback was already applied.");
  if (typeof e.key !== "string" || !Array.isArray(e.results) || !e.results.every(validResult)) {
    return no("That feedback isn't valid.");
  }
  const economy = getBalance(s.balance).economy;
  product.feedbackKeys = [...product.feedbackKeys, e.key].slice(-KEYS_KEPT);
  updateSatisfaction(s, economy, product.id, e.results);
  postBestAndWorst(s, economy, company, product, e.results);
  return OK;
}

function launchHeadline(company: CompanyState, product: ProductState): string {
  const scores = company.versions[product.versionId]?.scores ?? {};
  let best: BenchCategory | null = null;
  let top = -1;
  for (const c of BENCH_CATEGORIES) {
    const v = scores[c];
    if (typeof v === "number" && v > top) {
      best = c;
      top = v;
    }
  }
  const base = `${company.name} launches ${product.name}`;
  return best === null ? base : `${base}, scoring ${Math.round(top)} in ${SKILL_WORDS[best]}`;
}

function payingCustomers(s: WorldState, companyId: string): number {
  let n = 0;
  for (const region of REGION_IDS) {
    for (const group of GROUP_IDS) {
      n += customersOf(s.market, companyId, region, group, "personal");
      n += customersOf(s.market, companyId, region, group, "business");
    }
  }
  return n;
}

/**
 * The daily news step, after the market and the ledger: launches that went
 * live today (with their followers), customer milestones and the first paying
 * customer. Each milestone is told once; a big jump tells only the highest.
 */
export function newsDay(s: WorldState): void {
  const e = getBalance(s.balance).economy;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (!company) continue;
    for (const productId of Object.keys(company.products).sort()) {
      const product = company.products[productId];
      if (product?.status !== "live" || product.launchDay !== s.day) continue;
      company.fame.followers += e.fame.followersPerHype * product.hype;
      tellNews(s, company, "launch", launchHeadline(company, product), e.newsSentiment.launch);
    }
    const total = totalCustomers(s.market, id);
    const mark = CUSTOMER_MARKS.filter((m) => total >= m && m > company.announced.customers).at(-1);
    if (mark !== undefined) {
      company.announced.customers = mark;
      const headline = `${company.name} passes ${grouped(mark)} customers`;
      tellNews(s, company, "customers", headline, e.newsSentiment.customers);
    }
    if (!company.announced.firstPaying && payingCustomers(s, id) >= 1) {
      company.announced.firstPaying = true;
      const headline = `${company.name} gets its first paying customer`;
      tellNews(s, company, "firstPaying", headline, e.newsSentiment.firstPaying);
    }
  }
}
