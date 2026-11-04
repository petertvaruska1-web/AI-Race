import { Discovery, type DiscoveryKind } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { drawnAuthor, quotePost, tellNews } from "./feed.ts";
import type { CompanyState, DiscoveryState, SystemEvent, WorldState } from "./state.ts";

// Discoveries (game-layer design §10): what the scoring found in an AI, measured against
// the version it grew from and the world's other AIs, with one real question and reply as
// proof. Private until the player publishes one as news.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

type Headline = (company: string, label: string, about: string) => string;
const HEADLINES: Record<DiscoveryKind, Headline> = {
  strength: (c, l, a) => `${c}'s ${l} turns out to be surprisingly good at ${a}`,
  weakness: (c, l, a) => `${c} admits its ${l} struggles with ${a}`,
  better: (c, l, a) => `${c}'s ${l} got much better at ${a}`,
  worse: (c, l, a) => `${c}'s ${l} got worse at ${a}`,
  changed: (c, l, a) => `${c}'s ${l} has become more ${a}`,
};
const SENTIMENT: Record<DiscoveryKind, number> = {
  strength: 0.4,
  better: 0.4,
  changed: 0.2,
  weakness: 0.1,
  worse: 0.1,
};
const TONE = {
  strength: "praise",
  better: "praise",
  changed: "joke",
  weakness: "roast",
  worse: "roast",
} as const;

function ownerOf(s: WorldState, versionId: string): CompanyState | null {
  for (const id of Object.keys(s.companies).sort()) {
    const c = s.companies[id];
    if (c?.versions[versionId]) return c;
  }
  return null;
}

export function applyDiscoveries(
  s: WorldState,
  e: Extract<SystemEvent, { type: "discoveryFound" }>,
): Handled {
  const company = ownerOf(s, e.versionId);
  if (!company || company.rival) return no("Unknown version.");
  if (company.discoveries.some((d) => d.versionId === e.versionId)) return no("Already found.");
  const { perEval, kept } = getBalance(s.balance).economy.discoveries;
  const list: unknown[] = Array.isArray(e.discoveries) ? e.discoveries : [];
  const parsed = list.map((d) => Discovery.safeParse(d));
  if (list.length === 0 || list.length > perEval || !parsed.every((p) => p.success)) {
    return no("That discovery list isn't valid.");
  }
  parsed.forEach((p, k) => {
    if (!p.success) return;
    company.discoveries.push({
      ...p.data,
      id: `${e.versionId}-d${k}`,
      versionId: e.versionId,
      day: s.day,
      published: false,
    });
  });
  if (company.discoveries.length > kept) {
    company.discoveries.splice(0, company.discoveries.length - kept);
  }
  return OK;
}

/** The name the AI goes by in public: its product's, when one sells it, else its label. */
function publicName(company: CompanyState, d: DiscoveryState): string {
  const products = Object.values(company.products).filter((p) => p.versionId === d.versionId);
  const live = products.find((p) => p.status === "live") ?? products.at(-1);
  return live?.name ?? company.versions[d.versionId]?.label ?? "AI";
}

/** Publishing a finding: game-written news, plus a researcher's post quoting the proof. */
export function publishDiscovery(
  s: WorldState,
  company: CompanyState,
  discoveryId: string,
): Handled {
  const d = company.discoveries.find((x) => x.id === discoveryId);
  if (!d) return no("That discovery isn't yours.");
  if (d.published) return no("You've already published that.");
  const e = getBalance(s.balance).economy.discoveries;
  const label = company.versions[d.versionId]?.label ?? "AI";
  const sentiment = SENTIMENT[d.kind];
  tellNews(s, company, "discovery", HEADLINES[d.kind](company.name, label, d.about), sentiment);
  if (d.evidence) {
    const author = drawnAuthor(s, "researchers");
    const { prompt, reply } = d.evidence;
    quotePost(s, company, author, publicName(company, d), prompt, reply, TONE[d.kind], sentiment);
  }
  company.fame.followers *= 1 + e.publishFollowers;
  d.published = true;
  return OK;
}
