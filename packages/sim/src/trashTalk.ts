import { getBalance } from "./balance.ts";
import { tellNews } from "./feed.ts";
import { totalCustomers } from "./market.ts";
import { TRASH_TALK, type TrashEvent } from "./rivalCast.ts";
import { rngInt } from "./rng.ts";
import type { CompanyState, WorldState } from "./state.ts";

// The arch-rival's trash-talk (game-layer design §9): game text quoting its
// founder about something the player really did. Never an AI's words (R1), and
// rare: at most one line every `trashTalkGapDays` (C13).

/** The arch-rival comments on `event`, unless it spoke recently. */
export function trashTalk(
  s: WorldState,
  event: TrashEvent,
  detail: { player?: string; product?: string } = {},
): void {
  const arch = s.rivals ? s.companies[s.rivals.archId] : undefined;
  const rival = arch?.rival;
  if (!arch || !rival?.archetype) return;
  const gap = getBalance(s.balance).economy.rivals.trashTalkGapDays;
  if (rival.lastTrashTalkDay !== null && s.day - rival.lastTrashTalkDay < gap) return;
  const lines = TRASH_TALK[rival.archetype][event];
  const pick = rngInt(s.rng, lines.length);
  s.rng = pick.next;
  const line = (lines[pick.value] ?? lines[0])
    .replaceAll("{arch}", arch.name)
    .replaceAll("{player}", detail.player ?? "newcomers")
    .replaceAll("{product}", detail.product ?? "that AI");
  rival.lastTrashTalkDay = s.day;
  tellNews(s, arch, "trashTalk", `${arch.name}'s founder: "${line}"`, 0);
}

/** The daily trash-talk step, after the news: launches that went live today, and passing the arch-rival. */
export function trashTalkDay(s: WorldState): void {
  if (!s.rivals) return;
  const arch = s.companies[s.rivals.archId] as CompanyState;
  for (const p of Object.values(arch.products)) {
    if (p.status === "live" && p.launchDay === s.day)
      trashTalk(s, "archLaunch", { product: p.name });
  }
  const archCustomers = totalCustomers(s.market, arch.id);
  for (const id of Object.keys(s.companies).sort()) {
    const c = s.companies[id] as CompanyState;
    if (c.rival) continue;
    for (const p of Object.values(c.products)) {
      if (p.status === "live" && p.launchDay === s.day) {
        trashTalk(s, "playerLaunch", { player: c.name, product: p.name });
      }
    }
    if (!s.rivals.archPassed && archCustomers > 0 && totalCustomers(s.market, id) > archCustomers) {
      s.rivals.archPassed = true;
      trashTalk(s, "playerPassesArch", { player: c.name });
    }
  }
}
