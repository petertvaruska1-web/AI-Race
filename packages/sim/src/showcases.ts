import { type Command, GAME_NAMES } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { dateOfDay } from "./calendar.ts";
import { tellNews } from "./feed.ts";
import { book } from "./ledger.ts";
import { rngInt } from "./rng.ts";
import type { CompanyState, ShowcaseState, SystemEvent, WorldState } from "./state.ts";
import { formatMoney } from "./training.ts";

// Showcases (playtest pack §4.3): the player pays to play a Playground game live for
// the press. The worker plays it with the real AI; how far it got moves followers,
// reputation and the news.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

type StartShowcase = Extract<Command, { type: "startShowcase" }>;

export function showcaseCommand(
  s: WorldState,
  company: CompanyState,
  cmd: StartShowcase,
  newId: string | undefined,
): Handled {
  const b = getBalance(s.balance).economy.showcases;
  if (!newId) return no("Missing showcase id.");
  const product = Object.values(company.products).find(
    (p) => p.status === "live" && p.versionId === cmd.versionId,
  );
  if (!product) return no("Put that AI on sale before showing it off.");
  if (company.showcases.some((x) => x.status === "running")) {
    return no("Your last showcase is still going.");
  }
  const last = company.showcases.at(-1);
  if (last && s.day < last.day + b.cooldownDays) {
    return no(`The press saw you recently. Try again on ${dateOfDay(last.day + b.cooldownDays)}.`);
  }
  if (
    cmd.game === "story" &&
    typeof company.versions[cmd.versionId]?.scores?.creativity !== "number"
  ) {
    return no("Stories can't be judged on this computer yet.");
  }
  if (company.cash < b.fee) {
    return no(`A showcase costs ${formatMoney(b.fee)}. You have ${formatMoney(company.cash)}.`);
  }
  if (company.showcases.some((x) => x.id === newId)) return no("That showcase id is already used.");
  book(company, "events", b.fee);
  const seed = rngInt(s.rng, 2 ** 31);
  s.rng = seed.next;
  company.showcases.push({
    id: newId,
    day: s.day,
    game: cmd.game,
    difficulty: cmd.difficulty,
    settings: structuredClone(cmd.settings),
    productId: product.id,
    versionId: product.versionId,
    productName: product.name,
    seed: seed.value,
    status: "running",
    score: null,
    maxScore: null,
  });
  if (company.showcases.length > b.kept)
    company.showcases.splice(0, company.showcases.length - b.kept);
  return OK;
}

function findShowcase(
  s: WorldState,
  id: string,
): { company: CompanyState; showcase: ShowcaseState } | null {
  for (const cid of Object.keys(s.companies).sort()) {
    const company = s.companies[cid] as CompanyState;
    const showcase = company.showcases.find((x) => x.id === id);
    if (showcase) return { company, showcase };
  }
  return null;
}

/** The headline for a run that got `score` of `maxScore` of the way. */
export function showcaseHeadline(
  company: CompanyState,
  x: ShowcaseState,
  score: number,
  maxScore: number,
  stumble: number,
): string {
  const b = (n: number) => Math.round(n);
  const who = `${company.name}'s ${x.productName}`;
  const game = GAME_NAMES[x.game];
  const p = score / maxScore;
  if (p < stumble) return `${who} stumbles on the ${game} live on stage`;
  if (x.game === "story")
    return `${who} wins ${b(score)} of ${maxScore} audience points at Story Dice live on stage`;
  if (score >= maxScore) return `${who} clears the whole ${game} live on stage`;
  return `${who} reaches step ${b(score)} of the ${game} live on stage`;
}

/** Applies a showcase's result once: followers, reputation and news; or the refund. */
export function applyShowcase(
  s: WorldState,
  e: Extract<SystemEvent, { type: "showcaseResolved" }>,
): Handled {
  const found = findShowcase(s, e.showcaseId);
  if (found?.showcase.status !== "running") return no("This showcase is over.");
  const { company, showcase } = found;
  const b = getBalance(s.balance).economy.showcases;
  if ("failed" in e && e.failed) {
    showcase.status = "calledOff";
    book(company, "events", -b.fee);
    return OK;
  }
  const { score, maxScore } = e as { score: unknown; maxScore: unknown };
  if (
    typeof score !== "number" ||
    typeof maxScore !== "number" ||
    !Number.isFinite(score) ||
    !(maxScore > 0) ||
    score < 0 ||
    score > maxScore
  ) {
    return no("That showcase result isn't valid.");
  }
  showcase.status = "done";
  showcase.score = score;
  showcase.maxScore = maxScore;
  const mood = 2 * (score / maxScore) - 1;
  company.fame.followers *= 1 + b.followers * mood;
  tellNews(
    s,
    company,
    "showcase",
    showcaseHeadline(company, showcase, score, maxScore, b.stumble),
    mood,
  );
  return OK;
}
