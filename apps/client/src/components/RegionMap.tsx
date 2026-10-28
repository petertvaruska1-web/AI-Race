import { REGION_IDS, REGION_NAMES, type RegionId } from "@airace/protocol";
import { type ReactNode, useState } from "react";
import { REGION_CENTERS, STATE_TILES } from "../regions.ts";

export type RegionMarks = Partial<
  Record<RegionId, { campaign?: boolean; dataCenter?: "building" | "open" }>
>;

const GAP = 0.12;

/**
 * The nine regions as a tile map of the states (immersion design): each tinted
 * by its value (0..1). A region lifts and glows under the pointer with a readout;
 * campaigns pulse and data centers glow. Every region also has a real button
 * (shown as a legend, or hidden on the small map) that opens it.
 */
export function RegionMap({
  label,
  values,
  color,
  onOpen,
  readout,
  marks = {},
  small = false,
}: {
  label: string;
  values: Record<RegionId, number>;
  color: string;
  onOpen?: (region: RegionId, el: Element) => void;
  readout?: (region: RegionId) => ReactNode;
  marks?: RegionMarks;
  small?: boolean;
}) {
  const [hover, setHover] = useState<RegionId | null>(null);
  const at = hover ? REGION_CENTERS[hover] : null;
  return (
    <fieldset className="m-0 min-w-0 border-0 p-0" aria-label={label}>
      <div className="relative">
        <svg
          viewBox="-0.3 -0.3 12.6 8.6"
          className={`block w-full ${small ? "max-h-40" : ""}`}
          aria-hidden="true"
          onPointerLeave={() => setHover(null)}
        >
          {REGION_IDS.map((r) => (
            // biome-ignore lint/a11y/noStaticElementInteractions: a pointer shortcut; each region also has a real button below
            <g
              key={r}
              data-region={r}
              className={`region ${hover === r ? "region-hover" : ""} ${onOpen ? "cursor-pointer" : ""}`}
              onPointerEnter={() => setHover(r)}
              onClick={(e) => onOpen?.(r, e.currentTarget)}
            >
              {STATE_TILES.filter((t) => t.region === r).map((t) => (
                <rect
                  key={t.state}
                  data-state={t.state}
                  x={t.col + GAP / 2}
                  y={t.row + GAP / 2}
                  width={1 - GAP}
                  height={1 - GAP}
                  rx={0.18}
                  fill={color}
                  fillOpacity={0.12 + 0.88 * Math.min(1, Math.max(0, values[r]))}
                  stroke={color}
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
          ))}
          {REGION_IDS.map((r) => {
            const m = marks[r];
            const c = REGION_CENTERS[r];
            return (
              <g key={`${r}-marks`} className="pointer-events-none">
                {m?.campaign && (
                  <circle
                    cx={c.x}
                    cy={c.y}
                    r={0.9}
                    fill="none"
                    stroke={color}
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
        {hover && at && readout && (
          <div
            className="pointer-events-none absolute z-10 min-w-40 -translate-x-1/2 -translate-y-full rounded-lg border border-border bg-surface/95 px-3 py-2 text-xs shadow-[var(--glow)]"
            style={{
              left: `${((at.x + 0.3) / 12.6) * 100}%`,
              top: `${((at.y - 0.4 + 0.3) / 8.6) * 100}%`,
            }}
          >
            <p className="mb-1 font-semibold">{REGION_NAMES[hover]}</p>
            {readout(hover)}
          </div>
        )}
      </div>
      <ul className={small ? "sr-only" : "mt-3 flex flex-wrap gap-1.5"}>
        {REGION_IDS.map((r) => (
          <li key={r}>
            <button
              type="button"
              aria-label={`Open ${REGION_NAMES[r]}`}
              onClick={(e) => onOpen?.(r, e.currentTarget)}
              onPointerEnter={() => setHover(r)}
              onPointerLeave={() => setHover(null)}
              onFocus={() => setHover(r)}
              onBlur={() => setHover(null)}
              className={`chip transition ${hover === r ? "border-[var(--line)] text-text" : "text-muted"}`}
            >
              {REGION_NAMES[r]}
            </button>
          </li>
        ))}
      </ul>
    </fieldset>
  );
}
