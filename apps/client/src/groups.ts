import type { GroupId } from "@airace/protocol";

/** The customer groups in plain words (R7): one person, and the group. */
export const GROUPS: Record<GroupId, { one: string; many: string; wants: string }> = {
  students: { one: "Student", many: "Students", wants: "Knowledge and reasoning, at a low price" },
  creators: { one: "Creator", many: "Creators", wants: "Creativity and good writing" },
  developers: { one: "Developer", many: "Developers", wants: "Working code, fast" },
  businesses: {
    one: "Business",
    many: "Businesses",
    wants: "Consistent answers that follow instructions",
  },
  everyday: {
    one: "Everyday user",
    many: "Everyday users",
    wants: "Friendly chat, fast and cheap",
  },
  researchers: { one: "Researcher", many: "Researchers", wants: "Reasoning and knowledge" },
};
