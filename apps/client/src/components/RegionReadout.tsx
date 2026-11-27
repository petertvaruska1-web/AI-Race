import type { RegionId, WorldView } from "@airace/protocol";
import { formatCount } from "../format.ts";
import { leaderTops, regionCustomers } from "../regions.ts";

/**
 * What a region on the map tells (design §8.3), under its name: who leads it, in its map color,
 * and your customers there with your share of its willing people.
 */
export function RegionReadout({ view, region }: { view: WorldView; region: RegionId }) {
  const top = leaderTops(view)[region];
  const lead = view.companies.find((c) => c.id === top.leaderId);
  const n = regionCustomers(view, region);
  const people = view.world.market.regions[region].adoptable;
  return (
    <>
      <p
        className="player-text m-0 font-semibold"
        style={lead && top.color ? { color: top.color } : undefined}
      >
        {lead ? `Led by ${lead.name}` : "Nobody leads it yet"}
      </p>
      <p className="m-0 text-muted">
        <span className="tabular font-semibold text-text">{formatCount(n)}</span>
        {` customers of yours · ${people > 0 ? ((100 * n) / people).toFixed(2) : "0"}% of those willing`}
      </p>
    </>
  );
}
