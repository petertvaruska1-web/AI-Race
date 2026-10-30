import type { PublicCompany, WorldView } from "@airace/protocol";

/** The world's rivals, the arch-rival first, then by name. */
export function rivalCompanies(view: WorldView): PublicCompany[] {
  return view.companies
    .filter((c) => c.rival !== null)
    .sort(
      (a, b) =>
        Number(b.rival?.archetype != null) - Number(a.rival?.archetype != null) ||
        a.name.localeCompare(b.name),
    );
}

/** A rival AI on sale that the player can talk to or battle. */
export type RivalAi = {
  versionId: string;
  companyId: string;
  companyName: string;
  productId: string;
  productName: string;
};

export function rivalAis(view: WorldView): RivalAi[] {
  return rivalCompanies(view).flatMap((c) =>
    c.products
      .filter((p) => p.status === "live" && p.version)
      .map((p) => ({
        versionId: p.version?.id ?? "",
        companyId: c.id,
        companyName: c.name,
        productId: p.id,
        productName: p.name,
      })),
  );
}

/** Its share of the people willing to use an AI, in percent. */
export function shareOfWilling(view: WorldView, c: PublicCompany): number {
  const willing = view.world.market.adoptable;
  return willing > 0 ? (100 * c.customers) / willing : 0;
}
