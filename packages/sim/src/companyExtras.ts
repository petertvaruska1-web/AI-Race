import type { CompanyState } from "./state.ts";

// What sim-5 (the playtest pack) and sim-6 (Phase 4) add to every company, at their starting values.

/** The looks every company owns from the start. */
export const FREE_LOOKS = { orb: "orb:classic", logo: "logo:still" } as const;

export type CompanyExtras = Pick<
  CompanyState,
  "showcases" | "looks" | "sponsorships" | "apps" | "fund" | "loved" | "notebook" | "milestones"
>;

export function newCompanyExtras(): CompanyExtras {
  return {
    showcases: [],
    looks: { owned: [FREE_LOOKS.orb, FREE_LOOKS.logo], orb: FREE_LOOKS.orb, logo: FREE_LOOKS.logo },
    sponsorships: {},
    apps: {},
    fund: { balance: 0 },
    loved: 0,
    notebook: null,
    milestones: [],
  };
}
