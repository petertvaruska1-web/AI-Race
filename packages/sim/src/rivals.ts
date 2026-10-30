import {
  ARCHETYPE_STYLE,
  ARCHETYPE_WORDS,
  ARCHETYPES,
  type Archetype,
  RIVAL_STYLES,
  RivalPack,
  type RivalPackLineage,
  type RivalStyle,
} from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { newCompute } from "./compute.ts";
import { tellNews } from "./feed.ts";
import { newLedger } from "./ledger.ts";
import { newCompanyBusiness } from "./products.ts";
import { drawLook } from "./rivalCast.ts";
import { rngInt } from "./rng.ts";
import type { CompanyState, PackRelease, WorldState } from "./state.ts";

// Rivals (game-layer design §9): ordinary companies without an owner, so the
// market, money, fame and computers treat them like the player. They join from
// the rival pack's measurements, logged once as `rivalsJoined`.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

/** Rivals that join after this day announce themselves as newcomers. */
const EARLY_DAYS = 7;

export function isRival(company: CompanyState): boolean {
  return company.rival !== null;
}

/** Seats an arch-rival and three rivals from `pack`. Mutates `s`. */
export function joinRivals(s: WorldState, raw: unknown): Handled {
  if (s.phase !== "running") return no("The world is not running yet.");
  if (s.rivals !== null) return no("Rivals have already joined.");
  const parsed = RivalPack.safeParse(raw);
  if (!parsed.success) return no("The rival pack isn't valid.");
  const pack = parsed.data;
  const b = getBalance(s.balance);
  const r = b.economy.rivals;
  const draw = (n: number) => {
    const x = rngInt(s.rng, n);
    s.rng = x.next;
    return x.value;
  };
  const ofStyle = (style: RivalStyle) => pack.lineages.filter((l) => l.style === style);

  const archetype = ARCHETYPES[draw(ARCHETYPES.length)] as Archetype;
  const archStyle = ARCHETYPE_STYLE[archetype];
  const archLineage = ofStyle(archStyle)[draw(2)] as RivalPackLineage;
  const dropped = RIVAL_STYLES[draw(RIVAL_STYLES.length)];
  const cast: { lineage: RivalPackLineage; archetype: Archetype | null }[] = [
    { lineage: archLineage, archetype },
  ];
  for (const style of RIVAL_STYLES) {
    if (style === dropped) continue;
    const free = ofStyle(style).filter((l) => l.id !== archLineage.id);
    cast.push({ lineage: free[draw(free.length)] as RivalPackLineage, archetype: null });
  }

  const taken = new Set(Object.values(s.companies).map((c) => c.name.toLowerCase()));
  const primaries = new Set(Object.values(s.companies).map((c) => c.palette.primary.toLowerCase()));
  const ids: string[] = [];
  cast.forEach(({ lineage, archetype: type }, i) => {
    const drawn = drawLook(s.rng, type ?? lineage.style, taken, primaries);
    s.rng = drawn.rng;
    const id = `rival-${i + 1}`;
    const upcoming: PackRelease[] = [];
    let last = s.day;
    lineage.versions.forEach((v, k) => {
      const jitter = r.jitterDays > 0 ? draw(2 * r.jitterDays + 1) - r.jitterDays : 0;
      const planned = s.day + (r.firstRelease[i] ?? 0) + k * r.releaseGapDays + jitter;
      last = Math.max(last + 1, planned);
      upcoming.push({ ...v, releaseDay: last });
    });
    const cash = Math.round(b.foundingCash * r.startCash[lineage.style] * (type ? r.archBonus : 1));
    const company: CompanyState = {
      id,
      ownerUserId: null,
      startCash: cash,
      rival: {
        lineageId: lineage.id,
        style: lineage.style,
        focus: lineage.focus,
        archetype: type,
        productWord: drawn.look.productWord,
        upcoming,
        released: 0,
        lastReviewDay: s.day,
        lastTrashTalkDay: null,
        counters: { campaigns: 0, dataCenters: 0 },
      },
      name: drawn.look.name,
      logo: drawn.look.logo,
      palette: drawn.look.palette,
      cash,
      runs: {},
      versions: {},
      versionSeq: 0,
      ...newCompanyBusiness(b),
      ledger: newLedger(s.day),
      history: [],
      campaigns: {},
      compute: newCompute(b),
      dataCenters: {},
      reports: [],
      rescue: null,
      loan: null,
      ownership: 1,
    };
    s.companies[id] = company;
    ids.push(id);
  });

  const arch = s.companies[ids[0] as string] as CompanyState;
  s.rivals = { joinedDay: s.day, archId: arch.id };
  const headline =
    s.day <= EARLY_DAYS
      ? `${arch.name}, ${ARCHETYPE_WORDS[archetype]}, enters the AI race`
      : `New rivals enter the AI race, led by ${arch.name}`;
  tellNews(s, arch, "trashTalk", headline, 0);
  return OK;
}
