import type { PublicCompany, PublicVersion, WorldView } from "@airace/protocol";
import type { CardView } from "./store/cards.ts";

/** An AI another player shares with friends (Stage 2). */
export type SharedAi = {
  versionId: string;
  label: string;
  company: PublicCompany;
  version: PublicVersion;
  makerUserId: string | null;
};

/** Every other player's shared AIs, by company name then oldest first. */
export function sharedAis(view: WorldView): SharedAi[] {
  return view.companies
    .filter((c) => c.rival === null && c.id !== view.me.company?.id)
    .sort((a, b) => a.name.localeCompare(b.name))
    .flatMap((company) =>
      company.publishedVersions.map((version) => ({
        versionId: version.id,
        label: version.label,
        company,
        version,
        makerUserId: company.ownerUserId,
      })),
    );
}

/**
 * Whether a shared AI can answer now: with Fair mode off its maker's computer must be on (while
 * the cards aren't known yet, it is assumed on; the server has the last word).
 */
export function makerOnline(view: WorldView, cards: CardView[] | null, makerUserId: string | null) {
  if (view.world.fairMode || cards === null || makerUserId === null) return true;
  return cards.some((c) => c.userId === makerUserId && c.online);
}
