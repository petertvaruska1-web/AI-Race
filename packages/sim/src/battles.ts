import type { BattleSkill, Command } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { dateOfDay } from "./calendar.ts";
import { SKILL_WORDS, tellNews } from "./feed.ts";
import { book } from "./ledger.ts";
import { rngInt } from "./rng.ts";
import type {
  BattleRound,
  BattleState,
  CompanyState,
  ProductState,
  SystemEvent,
  WorldState,
} from "./state.ts";
import { formatMoney } from "./training.ts";
import { trashTalk } from "./trashTalk.ts";

// Battles (game-layer design §10): the player's AI and a rival's answer the same
// fresh challenges; real replies decide the result, which moves ranking points
// (an Elo rating per skill), followers and the news.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

type StartBattle = Extract<Command, { type: "startBattle" }>;

function liveRivalProduct(
  s: WorldState,
  productId: string,
): { rival: CompanyState; product: ProductState } | null {
  for (const id of Object.keys(s.companies).sort()) {
    const c = s.companies[id] as CompanyState;
    const p = c.products[productId];
    if (c.rival && p?.status === "live") return { rival: c, product: p };
  }
  return null;
}

/** Your live AI with the best score in `skill` (ties: the newest launch). */
function fighter(company: CompanyState, skill: BattleSkill): ProductState | null {
  let best: { p: ProductState; score: number } | null = null;
  for (const p of Object.values(company.products)) {
    if (p.status !== "live") continue;
    const score = company.versions[p.versionId]?.scores?.[skill] ?? -1;
    if (!best || score > best.score || (score === best.score && p.launchDay > best.p.launchDay)) {
      best = { p, score };
    }
  }
  return best?.p ?? null;
}

export function battleCommand(
  s: WorldState,
  company: CompanyState,
  cmd: StartBattle,
  newId: string | undefined,
): Handled {
  const b = getBalance(s.balance).economy.battles;
  if (!s.rivals) return no("There are no rivals in this world.");
  if (!newId) return no("Missing battle id.");
  const target = liveRivalProduct(s, cmd.rivalProductId);
  if (!target) return no("That AI isn't on sale.");
  const mine = fighter(company, cmd.skill);
  if (!mine) return no("Launch an AI before battling.");
  if (company.battles.some((x) => x.status === "running")) {
    return no("Your last battle is still going.");
  }
  const last = company.battles.filter((x) => x.rivalId === target.rival.id).at(-1);
  if (last && s.day < last.day + b.cooldownDays) {
    return no(
      `You battled ${target.rival.name} recently. Try again on ${dateOfDay(last.day + b.cooldownDays)}.`,
    );
  }
  const myVersion = company.versions[mine.versionId];
  const theirVersion = target.rival.versions[target.product.versionId];
  if (
    cmd.skill === "creativity" &&
    (typeof myVersion?.scores?.creativity !== "number" ||
      typeof theirVersion?.scores?.creativity !== "number")
  ) {
    return no("Creativity can't be judged on this computer yet.");
  }
  if (company.cash < b.fee) {
    return no(`A battle costs ${formatMoney(b.fee)}. You have ${formatMoney(company.cash)}.`);
  }
  if (company.battles.some((x) => x.id === newId)) return no("That battle id is already used.");
  book(company, "marketing", b.fee);
  const seed = rngInt(s.rng, 2 ** 31);
  s.rng = seed.next;
  company.battles.push({
    id: newId,
    day: s.day,
    skill: cmd.skill,
    rivalId: target.rival.id,
    mine: { productId: mine.id, versionId: mine.versionId, productName: mine.name },
    theirs: {
      productId: target.product.id,
      versionId: target.product.versionId,
      productName: target.product.name,
      packKey: theirVersion?.packKey ?? "",
    },
    seed: seed.value,
    status: "running",
    rounds: null,
    outcome: null,
  });
  if (company.battles.length > b.kept) company.battles.splice(0, company.battles.length - b.kept);
  return OK;
}

function findBattle(
  s: WorldState,
  battleId: string,
): { company: CompanyState; battle: BattleState } | null {
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id] as CompanyState;
    const battle = company.battles.find((x) => x.id === battleId);
    if (battle) return { company, battle };
  }
  return null;
}

function validRounds(raw: unknown, count: number): raw is BattleRound[] {
  if (!Array.isArray(raw) || raw.length !== count) return false;
  const answer = (a: unknown) => {
    const x = a as { reply?: unknown; score?: unknown } | null;
    return (
      typeof x?.reply === "string" &&
      typeof x.score === "number" &&
      Number.isFinite(x.score) &&
      x.score >= 0 &&
      x.score <= 1
    );
  };
  return raw.every((r) => {
    const x = r as { prompt?: unknown; mine?: unknown; theirs?: unknown } | null;
    return typeof x?.prompt === "string" && answer(x.mine) && answer(x.theirs);
  });
}

/** Elo's expected score for a player rated `a` against one rated `b`. */
export function expectedScore(a: number, b: number): number {
  return 1 / (1 + 10 ** ((b - a) / 400));
}

/** Applies a battle's result once: outcome, ranking points, followers, news; or the refund. */
export function applyBattle(
  s: WorldState,
  e: Extract<SystemEvent, { type: "battleResolved" }>,
): Handled {
  const found = findBattle(s, e.battleId);
  if (!found || found.battle.status !== "running") return no("This battle is over.");
  const { company, battle } = found;
  const b = getBalance(s.balance).economy.battles;
  if ("failed" in e && e.failed) {
    battle.status = "calledOff";
    book(company, "marketing", -b.fee);
    return OK;
  }
  const raw = (e as { rounds?: unknown }).rounds;
  if (!validRounds(raw, b.rounds)) return no("That battle result isn't valid.");
  battle.rounds = raw.map((r) => ({
    prompt: r.prompt,
    mine: { reply: r.mine.reply, score: r.mine.score },
    theirs: { reply: r.theirs.reply, score: r.theirs.score },
  }));
  let wins = 0;
  let losses = 0;
  let total = 0;
  for (const r of raw) {
    if (r.mine.score > r.theirs.score) wins += 1;
    if (r.mine.score < r.theirs.score) losses += 1;
    total += r.mine.score - r.theirs.score;
  }
  const outcome = total > b.drawMargin ? "win" : total < -b.drawMargin ? "loss" : "draw";
  battle.status = "done";
  battle.outcome = outcome;

  const rival = s.companies[battle.rivalId] as CompanyState;
  const table = s.points[battle.skill] ?? {};
  const mine = table[company.id] ?? b.pointsStart;
  const theirs = table[rival.id] ?? b.pointsStart;
  const score = outcome === "win" ? 1 : outcome === "loss" ? 0 : 0.5;
  table[company.id] = Math.round(mine + b.pointsK * (score - expectedScore(mine, theirs)));
  table[rival.id] = Math.round(theirs + b.pointsK * (1 - score - expectedScore(theirs, mine)));
  s.points[battle.skill] = table;

  if (outcome !== "draw") {
    const [winner, loser] = outcome === "win" ? [company, rival] : [rival, company];
    winner.fame.followers *= 1 + b.followers.win;
    loser.fame.followers *= 1 + b.followers.loss;
  }
  const skill = SKILL_WORDS[battle.skill];
  const me = `${company.name}'s ${battle.mine.productName}`;
  const them = `${rival.name}'s ${battle.theirs.productName}`;
  const headline =
    outcome === "win"
      ? `${me} beats ${them} in a ${skill} battle, ${wins}–${losses}`
      : outcome === "loss"
        ? `${them} beats ${me} in a ${skill} battle, ${losses}–${wins}`
        : `${me} and ${them} draw in a ${skill} battle, ${wins}–${losses}`;
  tellNews(s, company, "battle", headline, b.sentiment[outcome]);
  if (outcome === "win" && rival.id === s.rivals?.archId) {
    trashTalk(s, "playerWinsBattle", { player: company.name });
  }
  return OK;
}
