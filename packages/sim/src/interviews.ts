import { getBalance } from "./balance.ts";
import { dateOfDay } from "./calendar.ts";
import { quotePost, tellNews, tellViral } from "./feed.ts";
import { findProduct } from "./products.ts";
import { rngInt } from "./rng.ts";
import type { InterviewAnswer, PostItem, SystemEvent, WorldState } from "./state.ts";

// Interviews (game-layer design §10): the player interviews their AI in front of the press
// and publishes the answers they choose. The server holds the conversation, so only what the
// AI really said is published (R1); the world gets the answers with their scores.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

export const OUTLETS = [
  { name: "The Daily Byte", handle: "@dailybyte" },
  { name: "Circuit Weekly", handle: "@circuitweekly" },
  { name: "TechTalk Radio", handle: "@techtalkradio" },
  { name: "The Model Times", handle: "@modeltimes" },
] as const;

/** Measured personality words that charm the press, and ones that put it off. */
const CHARMING = ["warm", "bold", "inventive"];
const OFF_PUTTING = ["slip", "timid"];
const TRAIT_BONUS = 0.05;

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

function validAnswer(a: unknown): a is InterviewAnswer {
  if (typeof a !== "object" || a === null) return false;
  const x = a as Record<string, unknown>;
  return (
    typeof x.question === "string" &&
    typeof x.reply === "string" &&
    typeof x.checked === "boolean" &&
    typeof x.score === "number" &&
    x.score >= 0 &&
    x.score <= 1
  );
}

/** How the AI's measured personality moves the press: +0.05 or −0.05 per word. */
export function traitBonus(words: readonly string[]): number {
  let bonus = 0;
  for (const w of words) {
    if (CHARMING.some((c) => w.includes(c))) bonus += TRAIT_BONUS;
    if (OFF_PUTTING.some((c) => w.includes(c))) bonus -= TRAIT_BONUS;
  }
  return bonus;
}

export function applyInterview(
  s: WorldState,
  e: Extract<SystemEvent, { type: "interviewScored" }>,
): Handled {
  const found = findProduct(s, e.productId);
  if (!found || found.company.rival) return no("That AI isn't one of yours.");
  const { company, product } = found;
  if (product.status !== "live") return no("That AI isn't on sale.");
  const b = getBalance(s.balance).economy.interviews;
  const answers: unknown[] = Array.isArray(e.answers) ? e.answers : [];
  if (answers.length === 0 || answers.length > b.maxAnswers || !answers.every(validAnswer)) {
    return no("That interview isn't valid.");
  }
  if (typeof e.interviewId !== "string" || company.interviews.some((i) => i.id === e.interviewId)) {
    return no("Already published.");
  }
  const last = company.lastInterviewDay;
  if (last !== null && s.day < last + b.cooldownDays) {
    return no(`You gave an interview recently. Next one on ${dateOfDay(last + b.cooldownDays)}.`);
  }
  const list = answers as InterviewAnswer[];
  const mean = list.reduce((sum, a) => sum + a.score, 0) / list.length;
  const words = company.versions[product.versionId]?.personalityWords ?? [];
  const r = clamp(mean + traitBonus(words), 0, 1);
  const pick = rngInt(s.rng, OUTLETS.length);
  s.rng = pick.next;
  const outlet = OUTLETS[pick.value] ?? OUTLETS[0];

  const headline =
    r >= 0.7
      ? `${product.name} wows in its interview with ${outlet.name}`
      : r >= 0.4
        ? `${product.name} gives a so-so interview to ${outlet.name}`
        : `${product.name} stumbles through an interview with ${outlet.name}`;
  tellNews(s, company, "interview", headline, clamp(2 * r - 1, -0.8, 0.8));

  let best = list[0] as InterviewAnswer;
  let worst = best;
  for (const a of list) {
    if (a.score > best.score) best = a;
    if (a.score < worst.score) worst = a;
  }
  const author: PostItem["author"] = {
    name: outlet.name,
    handle: outlet.handle,
    group: "everyday",
    spotlightId: null,
  };
  const quote = (a: InterviewAnswer, tone: "praise" | "roast") =>
    quotePost(s, company, author, product.name, a.question, a.reply, tone, 2 * a.score - 1, true);
  quote(best, "praise");
  if (worst !== best && (r < 0.4 || worst.score <= b.viralBelow)) {
    const roast = quote(worst, "roast");
    if (worst.score <= b.viralBelow && company.fame.followers >= b.viralFollowers) {
      tellViral(s, company, roast, product.name);
    }
  }
  company.fame.followers *= 1 + b.followersPerPoint * (2 * r - 1);
  company.interviews.push({
    id: e.interviewId,
    day: s.day,
    productId: product.id,
    productName: product.name,
    outlet: outlet.name,
    mean,
    answers: list.map((a) => ({ ...a })),
  });
  if (company.interviews.length > b.kept) {
    company.interviews.splice(0, company.interviews.length - b.kept);
  }
  company.lastInterviewDay = s.day;
  return OK;
}
