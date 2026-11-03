import { getBalance } from "./balance.ts";
import type { SystemEvent, WorldState } from "./state.ts";

// The player's notebook (game-layer design §10): text the server keeps for this company's
// training only. The world holds its sha256 and word count, never the text (R2).

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

const SHA256 = /^[0-9a-f]{64}$/;

export function applyNotebookSaved(
  s: WorldState,
  e: Extract<SystemEvent, { type: "notebookSaved" }>,
): Handled {
  const company = s.companies[e.companyId];
  if (!company || company.rival) return no("Unknown company.");
  const max = getBalance(s.balance).economy.notebook.maxWords;
  if (typeof e.hash !== "string" || !SHA256.test(e.hash)) return no("That notebook isn't valid.");
  if (!Number.isInteger(e.words) || e.words < 0 || e.words > max) {
    return no("That notebook isn't valid.");
  }
  company.notebook = { hash: e.hash, words: e.words, savedDay: s.day };
  return OK;
}
