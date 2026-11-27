import { REGION_IDS, type RegionId } from "@airace/protocol";
import { REGION_CENTERS, type RegionMarks, type RegionTop, STATE_TILES } from "../regions.ts";

const GAP = 0.12;
/** Plain glass on the flat map: a cool gray. */
const PLAIN_GLASS = "#8fa3bd";

/**
 * The map without 3D (design §8.0's flat stand-in): the 50 states and DC as tiles on a 12 × 8
 * grid, each region in its top's color and the more solid the higher it stands, plain glass where
 * nobody leads. A region lights under the pointer and opens on a click; the player's marks stand
 * on it.
 */
export function TileMap({
  tops,
  marks,
  focus,
  hover,
  onHover,
  onOpen,
  small,
  markColor,
}: {
  tops: Record<RegionId, RegionTop>;
  marks: RegionMarks;
  focus: RegionId | null;
  hover: RegionId | null;
  onHover: (region: RegionId | null) => void;
  onOpen?: (region: RegionId, el: Element) => void;
  small: boolean;
  markColor: string;
}) {
  return (
    <svg
      viewBox="-0.3 -0.3 12.6 8.6"
      className={`mx-auto block h-full w-full ${small ? "max-h-40" : ""}`}
      aria-hidden="true"
      onPointerLeave={() => onHover(null)}
    >
      {REGION_IDS.map((r) => {
        const top = tops[r];
        const fill = top.color ?? PLAIN_GLASS;
        return (
          // biome-ignore lint/a11y/noStaticElementInteractions: a pointer shortcut; each region also has a real key
          <g
            key={r}
            data-region={r}
            opacity={focus && focus !== r ? 0.35 : 1}
            className={`region ${hover === r ? "region-hover" : ""} ${onOpen ? "cursor-pointer" : ""}`}
            onPointerEnter={() => onHover(r)}
            onClick={(e) => onOpen?.(r, e.currentTarget)}
          >
            {STATE_TILES.filter((t) => t.region === r).map((t) => (
              <rect
                key={t.state}
                data-state={t.state}
                data-plain={top.color ? undefined : "true"}
                x={t.col + GAP / 2}
                y={t.row + GAP / 2}
                width={1 - GAP}
                height={1 - GAP}
                rx={0.18}
                fill={fill}
                fillOpacity={top.color ? 0.3 + 0.7 * top.height : 0.14}
                stroke={fill}
                strokeOpacity={hover === r ? 0.9 : 0.25}
                strokeWidth={0.04}
                className="region-tile"
              />
            ))}
            {!small &&
              STATE_TILES.filter((t) => t.region === r).map((t) => (
                <text
                  key={`${t.state}-label`}
                  x={t.col + 0.5}
                  y={t.row + 0.62}
                  textAnchor="middle"
                  className="pointer-events-none fill-[var(--text)] text-[0.3px] opacity-60"
                >
                  {t.state}
                </text>
              ))}
          </g>
        );
      })}
      {REGION_IDS.map((r) => {
        const m = marks[r];
        const c = REGION_CENTERS[r];
        return (
          <g key={`${r}-marks`} className="pointer-events-none">
            {m?.sponsor && (
              <text x={c.x + 0.35} y={c.y - 0.35} fontSize={0.7} aria-label="Sponsored event">
                {m.sponsor}
              </text>
            )}
            {m?.campaign && (
              <circle
                cx={c.x}
                cy={c.y}
                r={0.9}
                fill="none"
                stroke={markColor}
                strokeWidth={0.08}
                className="map-pulse"
              />
            )}
            {m?.dataCenter === "open" && (
              <rect
                x={c.x - 0.22}
                y={c.y - 0.22}
                width={0.44}
                height={0.44}
                rx={0.08}
                fill="var(--text)"
                className="map-glow"
              />
            )}
            {m?.dataCenter === "building" && (
              <circle
                cx={c.x}
                cy={c.y}
                r={0.32}
                fill="none"
                stroke="var(--text)"
                strokeWidth={0.07}
                strokeDasharray="0.15 0.12"
                className="map-spin"
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
