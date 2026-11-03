import { type Command, REGION_NAMES, type SponsorKind } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { tellNews } from "./feed.ts";
import { book } from "./ledger.ts";
import { awarenessOf, setAwareness } from "./market.ts";
import type { CompanyState, WorldState } from "./state.ts";
import { formatMoney } from "./training.ts";

// Sponsorships (playtest pack §5): the company's name on a big event in a region. Each
// day it runs, the event's crowd (its customer groups there) hears more of the company.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

const KIND_WORDS: Record<SponsorKind, string> = {
  stadium: "a stadium",
  conference: "a tech conference",
  festival: "an arts festival",
};

type Sponsor = Extract<Command, { type: "sponsor" }>;

export function sponsorCommand(
  s: WorldState,
  company: CompanyState,
  cmd: Sponsor,
  newId: string | undefined,
): Handled {
  if (!newId) return no("Missing sponsorship id.");
  const kind = getBalance(s.balance).economy.sponsorships.kinds[cmd.kind];
  const running = Object.values(company.sponsorships).some(
    (x) => x.region === cmd.region && x.kind === cmd.kind && s.day < x.endDay,
  );
  if (running) return no(`You already sponsor ${KIND_WORDS[cmd.kind]} there.`);
  if (company.cash < kind.price) {
    return no(
      `That sponsorship costs ${formatMoney(kind.price)}. You have ${formatMoney(company.cash)}.`,
    );
  }
  if (company.sponsorships[newId]) return no("That sponsorship id is already used.");
  book(company, "sponsorships", kind.price);
  company.sponsorships[newId] = {
    id: newId,
    region: cmd.region,
    kind: cmd.kind,
    startDay: s.day,
    endDay: s.day + kind.days,
  };
  tellNews(
    s,
    company,
    "sponsorship",
    `${company.name} puts its name on ${KIND_WORDS[cmd.kind]} in the ${REGION_NAMES[cmd.region]}`,
    0.3,
  );
  return OK;
}

/** A day of every running sponsorship: its crowd's awareness rises. Call before the market step. */
export function sponsorshipsDay(s: WorldState): void {
  const kinds = getBalance(s.balance).economy.sponsorships.kinds;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id] as CompanyState;
    for (const x of Object.values(company.sponsorships)) {
      if (s.day < x.startDay || s.day >= x.endDay) continue;
      const kind = kinds[x.kind];
      for (const group of kind.groups) {
        const a = awarenessOf(s.market, id, x.region, group);
        setAwareness(s.market, id, x.region, group, a + (1 - a) * kind.rate);
      }
    }
  }
}
