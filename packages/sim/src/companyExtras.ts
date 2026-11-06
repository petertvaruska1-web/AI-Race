import { DEPARTMENTS } from "@airace/protocol";
import type { CompanyState } from "./state.ts";

// What sim-5 (the playtest pack) and sim-6 (Phase 4) add to every company, at their starting values.

/** The looks every company owns from the start. */
export const FREE_LOOKS = { orb: "orb:classic", logo: "logo:still" } as const;

export type CompanyExtras = Pick<
  CompanyState,
  | "showcases"
  | "looks"
  | "sponsorships"
  | "apps"
  | "fund"
  | "departments"
  | "loved"
  | "lessons"
  | "notebook"
  | "milestones"
  | "discoveries"
  | "interviews"
  | "lastInterviewDay"
>;

export function newCompanyExtras(): CompanyExtras {
  return {
    showcases: [],
    looks: { owned: [FREE_LOOKS.orb, FREE_LOOKS.logo], orb: FREE_LOOKS.orb, logo: FREE_LOOKS.logo },
    sponsorships: {},
    apps: {},
    fund: { balance: 0 },
    departments: newDepartments(),
    loved: 0,
    lessons: 0,
    notebook: null,
    milestones: [],
    discoveries: [],
    interviews: [],
    lastInterviewDay: null,
  };
}

/** Every department unfunded, with no team. */
export function newDepartments(): CompanyState["departments"] {
  return Object.fromEntries(
    DEPARTMENTS.map((d) => [d, { budget: 0, staffed: 0 }]),
  ) as CompanyState["departments"];
}
