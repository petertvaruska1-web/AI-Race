import { REGION_IDS, type RegionId, type WorldView } from "@airace/protocol";
import { getBalance } from "@airace/sim";
import { readableOn } from "./theme/glass.ts";

/** A region's top on the map: its color (null: plain glass), its height (0..1) and whose it is. */
export type RegionTop = { color: string | null; height: number; leaderId: string | null };

/** The player's own marks on the map (R2: only their own campaigns, data centers and events). */
export type RegionMarks = Partial<
  Record<RegionId, { campaign?: boolean; dataCenter?: "building" | "open"; sponsor?: string }>
>;

const PLAIN: RegionTop = { color: null, height: 0, leaderId: null };

type Holder = { share: number; color: string; id: string };

/**
 * Tops from each region's holder: raised 0.25..1 by its share of the region's willing people
 * against the largest such share, in its color made readable in the map's dark display (the map
 * sits in one on either glass, so it looks the same for everybody); plain glass where nobody is.
 */
function topsOf(holders: Record<RegionId, Holder | null>): Record<RegionId, RegionTop> {
  const most = Math.max(0, ...REGION_IDS.map((r) => holders[r]?.share ?? 0));
  return Object.fromEntries(
    REGION_IDS.map((r) => {
      const h = holders[r];
      if (!h || most <= 0) return [r, { ...PLAIN }];
      return [
        r,
        {
          color: readableOn(h.color, "dark"),
          height: 0.25 + (0.75 * h.share) / most,
          leaderId: h.id,
        },
      ];
    }),
  ) as Record<RegionId, RegionTop>;
}

/**
 * Who leads each region (design §8.3): the company with the most customers there (ties to the
 * lower id), from public data only, so every player sees the same map. Nobody leads a region
 * without customers.
 */
export function leaderTops(view: WorldView): Record<RegionId, RegionTop> {
  const holders = {} as Record<RegionId, Holder | null>;
  for (const r of REGION_IDS) {
    const people = view.world.market.regions[r].adoptable;
    const top = [...view.companies].sort(
      (a, b) => b.customersByRegion[r] - a.customersByRegion[r] || a.id.localeCompare(b.id),
    )[0];
    const n = top?.customersByRegion[r] ?? 0;
    holders[r] =
      top && n > 0 && people > 0
        ? { share: n / people, color: top.palette.primary, id: top.id }
        : null;
  }
  return topsOf(holders);
}

/** One company's own regions (its window's map): plain glass where it has no customers. */
export function companyTops(view: WorldView, companyId: string): Record<RegionId, RegionTop> {
  const company = view.companies.find((c) => c.id === companyId);
  const holders = {} as Record<RegionId, Holder | null>;
  for (const r of REGION_IDS) {
    const people = view.world.market.regions[r].adoptable;
    const n = company?.customersByRegion[r] ?? 0;
    holders[r] =
      company && n > 0 && people > 0
        ? { share: n / people, color: company.palette.primary, id: company.id }
        : null;
  }
  return topsOf(holders);
}

/**
 * The 50 states and DC as tiles on a 12 x 8 grid (a tile-grid map of the US),
 * each in its census division: the game's nine regions.
 */
export const STATE_TILES: { state: string; col: number; row: number; region: RegionId }[] = [
  { state: "AK", col: 0, row: 0, region: "pacific" },
  { state: "ME", col: 11, row: 0, region: "newEngland" },
  { state: "WI", col: 6, row: 1, region: "greatLakes" },
  { state: "VT", col: 10, row: 1, region: "newEngland" },
  { state: "NH", col: 11, row: 1, region: "newEngland" },
  { state: "WA", col: 1, row: 2, region: "pacific" },
  { state: "ID", col: 2, row: 2, region: "mountain" },
  { state: "MT", col: 3, row: 2, region: "mountain" },
  { state: "ND", col: 4, row: 2, region: "plains" },
  { state: "MN", col: 5, row: 2, region: "plains" },
  { state: "IL", col: 6, row: 2, region: "greatLakes" },
  { state: "MI", col: 7, row: 2, region: "greatLakes" },
  { state: "NY", col: 9, row: 2, region: "midAtlantic" },
  { state: "MA", col: 10, row: 2, region: "newEngland" },
  { state: "OR", col: 1, row: 3, region: "pacific" },
  { state: "NV", col: 2, row: 3, region: "mountain" },
  { state: "WY", col: 3, row: 3, region: "mountain" },
  { state: "SD", col: 4, row: 3, region: "plains" },
  { state: "IA", col: 5, row: 3, region: "plains" },
  { state: "IN", col: 6, row: 3, region: "greatLakes" },
  { state: "OH", col: 7, row: 3, region: "greatLakes" },
  { state: "PA", col: 8, row: 3, region: "midAtlantic" },
  { state: "NJ", col: 9, row: 3, region: "midAtlantic" },
  { state: "CT", col: 10, row: 3, region: "newEngland" },
  { state: "RI", col: 11, row: 3, region: "newEngland" },
  { state: "CA", col: 1, row: 4, region: "pacific" },
  { state: "UT", col: 2, row: 4, region: "mountain" },
  { state: "CO", col: 3, row: 4, region: "mountain" },
  { state: "NE", col: 4, row: 4, region: "plains" },
  { state: "MO", col: 5, row: 4, region: "plains" },
  { state: "KY", col: 6, row: 4, region: "midSouth" },
  { state: "WV", col: 7, row: 4, region: "southAtlantic" },
  { state: "VA", col: 8, row: 4, region: "southAtlantic" },
  { state: "MD", col: 9, row: 4, region: "southAtlantic" },
  { state: "DE", col: 10, row: 4, region: "southAtlantic" },
  { state: "AZ", col: 2, row: 5, region: "mountain" },
  { state: "NM", col: 3, row: 5, region: "mountain" },
  { state: "KS", col: 4, row: 5, region: "plains" },
  { state: "AR", col: 5, row: 5, region: "southCentral" },
  { state: "TN", col: 6, row: 5, region: "midSouth" },
  { state: "NC", col: 7, row: 5, region: "southAtlantic" },
  { state: "SC", col: 8, row: 5, region: "southAtlantic" },
  { state: "DC", col: 9, row: 5, region: "southAtlantic" },
  { state: "OK", col: 4, row: 6, region: "southCentral" },
  { state: "LA", col: 5, row: 6, region: "southCentral" },
  { state: "MS", col: 6, row: 6, region: "midSouth" },
  { state: "AL", col: 7, row: 6, region: "midSouth" },
  { state: "GA", col: 8, row: 6, region: "southAtlantic" },
  { state: "HI", col: 0, row: 7, region: "pacific" },
  { state: "TX", col: 4, row: 7, region: "southCentral" },
  { state: "FL", col: 9, row: 7, region: "southAtlantic" },
];

/** Where a region's label and marks sit: the middle of its tiles on the mainland. */
export const REGION_CENTERS: Record<RegionId, { x: number; y: number }> = Object.fromEntries(
  REGION_IDS.map((r) => {
    // Alaska and Hawaii would pull the Pacific's middle out to sea.
    const tiles = STATE_TILES.filter((t) => t.region === r && t.state !== "AK" && t.state !== "HI");
    const x = tiles.reduce((n, t) => n + t.col + 0.5, 0) / tiles.length;
    const y = tiles.reduce((n, t) => n + t.row + 0.5, 0) / tiles.length;
    return [r, { x, y }];
  }),
) as Record<RegionId, { x: number; y: number }>;

/**
 * The player's share of each region's willing people, spread from the weakest
 * region (0.25) to the strongest (1), so differences show even when shares are close.
 */
export function shareByRegion(view: WorldView): Record<RegionId, number> {
  const mine = view.me.company?.customers.byRegion;
  const share = REGION_IDS.map((r) => {
    const people = view.world.market.regions[r].adoptable;
    const n = mine ? Object.values(mine[r]).reduce((a, b) => a + b, 0) : 0;
    return people > 0 ? n / people : 0;
  });
  const top = Math.max(...share);
  const low = Math.min(...share);
  const spread = (x: number) =>
    top <= 0 ? 0 : top === low ? 1 : 0.25 + (0.75 * (x - low)) / (top - low);
  return Object.fromEntries(REGION_IDS.map((r, i) => [r, spread(share[i] ?? 0)])) as Record<
    RegionId,
    number
  >;
}

/** The player's customers in a region, over every group. */
export function regionCustomers(view: WorldView, region: RegionId): number {
  const byGroup = view.me.company?.customers.byRegion[region];
  return byGroup ? Object.values(byGroup).reduce((a, b) => a + b, 0) : 0;
}

/** How fast answers come in a region, in words, from the player's open data centers. */
export function closenessWords(view: WorldView, region: RegionId): string {
  const neighbors = getBalance(view.world.balance).economy.neighbors;
  const open = (view.me.company?.dataCenters ?? []).filter((d) => d.open);
  if (open.some((d) => d.region === region)) return "Fast answers: a data center is here";
  if (open.some((d) => neighbors[d.region].includes(region))) {
    return "Fast answers: a data center is nearby";
  }
  return "Answers come from the cloud";
}
