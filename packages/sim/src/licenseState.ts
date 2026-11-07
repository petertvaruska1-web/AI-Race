import type { LicensesState } from "./state.ts";

/** A world with no license offers, proposals or deals. */
export function newLicenses(): LicensesState {
  return { offers: [], proposals: [], deals: [], seq: 0, lastRequest: {} };
}
