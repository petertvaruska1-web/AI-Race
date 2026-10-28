import { type RefObject, useEffect, useRef, useState } from "react";
import { dayLabel, formatCompact } from "../../format.ts";

export type DaySeries = { id: string; label: string; color: string; values: number[] };

const RANGES = [
  { id: 30, label: "30 days" },
  { id: 90, label: "90 days" },
  { id: 0, label: "All" },
] as const;
const PAD = { left: 52, right: 14, top: 14, bottom: 26 };

/** The box's width in pixels, kept current as it resizes (a fallback where it can't be measured). */
function useWidth(box: RefObject<HTMLElement | null>, fallback: number): number {
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const measure = () => setWidth(el.clientWidth || fallback);
    measure();
    if (typeof ResizeObserver !== "function") return;
    const watch = new ResizeObserver(measure);
    watch.observe(el);
    return () => watch.disconnect();
  }, [box, fallback]);
  return width;
}

/** Round steps for an axis: 1, 2 or 5 times a power of ten. */
function ticks(lo: number, hi: number, count: number): number[] {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out;
}

/**
 * A chart of numbers day by day (immersion design): lines, or areas stacked
 * one on another, over the last 30, 90 or all days. The pointer reads out
 * every series for its day. Lines draw themselves in when shown.
 */
export function DayChart({
  label,
  days,
  series,
  format,
  axis = formatCompact,
  stacked = false,
  height = 240,
}: {
  label: string;
  days: number[];
  series: DaySeries[];
  /** For the readout. */
  format: (n: number) => string;
  /** For the axis: short. */
  axis?: (n: number) => string;
  stacked?: boolean;
  height?: number;
}) {
  const [range, setRange] = useState<number>(0);
  const [hover, setHover] = useState<number | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const width = useWidth(box, 640);

  if (days.length === 0) {
    return <p className="text-sm text-muted">The day-by-day history starts tomorrow.</p>;
  }

  const n = range > 0 ? Math.min(days.length, range) : days.length;
  const shownDays = days.slice(-n);
  const shown = series.map((s) => ({ ...s, values: s.values.slice(-n) }));
  // Each series' top edge: its own values, or the running total when stacked.
  const tops: number[][] = [];
  for (const [k, s] of shown.entries()) {
    tops.push(s.values.map((v, i) => v + (stacked && k > 0 ? (tops[k - 1]?.[i] ?? 0) : 0)));
  }
  const all = tops.flat();
  const lo = stacked ? 0 : Math.min(0, ...all);
  const hi = Math.max(lo + 1, ...all) * 1.08;
  const plotW = Math.max(1, width - PAD.left - PAD.right);
  const plotH = height - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (n <= 1 ? plotW / 2 : (i * plotW) / (n - 1));
  const y = (v: number) => PAD.top + (1 - (v - lo) / (hi - lo)) * plotH;
  const line = (vals: number[]) => vals.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("");
  const area = (top: number[], below: number[] | null) =>
    `${line(top)}L${x(n - 1)},${y(below?.[n - 1] ?? lo)}${(below ?? top.map(() => lo))
      .map((_, j) => {
        const i = n - 1 - j;
        return `L${x(i)},${y(below?.[i] ?? lo)}`;
      })
      .join("")}Z`;
  const yTicks = ticks(lo, hi, 4);
  const xEvery = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(plotW / 90))));

  function onMove(e: React.PointerEvent<SVGRectElement>) {
    const left = e.currentTarget.ownerSVGElement?.getBoundingClientRect().left ?? 0;
    const i = n <= 1 ? 0 : Math.round(((e.clientX - left - PAD.left) / plotW) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  }

  const at = hover === null ? null : Math.min(hover, n - 1);
  return (
    <div>
      {days.length > 30 && (
        <fieldset className="m-0 mb-3 flex gap-1 border-0 p-0" aria-label="How far back">
          {RANGES.map((r) => (
            <button
              key={r.id}
              type="button"
              aria-pressed={range === r.id}
              onClick={() => setRange(r.id)}
              className={`rounded-md px-2.5 py-1 text-xs transition ${range === r.id ? "bg-brand text-on-brand" : "bg-surface-2 text-muted hover:text-text"}`}
            >
              {r.label}
            </button>
          ))}
        </fieldset>
      )}
      <div ref={box} className="relative" role="img" aria-label={label} data-points={n}>
        <svg width={width} height={height} className="block overflow-visible" aria-hidden="true">
          {yTicks.map((v) => (
            <g key={v}>
              <line
                x1={PAD.left}
                x2={width - PAD.right}
                y1={y(v)}
                y2={y(v)}
                stroke="var(--border)"
                strokeDasharray={v === 0 ? undefined : "3 5"}
              />
              <text
                x={PAD.left - 8}
                y={y(v) + 4}
                textAnchor="end"
                className="fill-[var(--muted)] text-[11px]"
              >
                {axis(v)}
              </text>
            </g>
          ))}
          {shownDays.map((d, i) =>
            i % xEvery === 0 ? (
              <text
                key={d}
                x={x(i)}
                y={height - 6}
                textAnchor="middle"
                className="fill-[var(--muted)] text-[11px]"
              >
                {dayLabel(d)}
              </text>
            ) : null,
          )}
          <g key={`${n}-${shown.length}`}>
            {shown.map((s, k) => (
              <g key={s.id}>
                <path
                  d={area(tops[k] ?? [], stacked && k > 0 ? (tops[k - 1] ?? null) : null)}
                  fill={s.color}
                  opacity={stacked ? 0.55 : 0.14}
                  className="chart-fade"
                />
                <path
                  d={line(tops[k] ?? [])}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={2}
                  pathLength={1}
                  className="chart-draw"
                />
              </g>
            ))}
          </g>
          {at !== null && (
            <g>
              <line
                x1={x(at)}
                x2={x(at)}
                y1={PAD.top}
                y2={PAD.top + plotH}
                stroke="var(--text)"
                strokeOpacity={0.35}
              />
              {shown.map((s, k) => (
                <circle
                  key={s.id}
                  cx={x(at)}
                  cy={y(tops[k]?.[at] ?? 0)}
                  r={4}
                  fill={s.color}
                  stroke="var(--bg)"
                  strokeWidth={2}
                />
              ))}
            </g>
          )}
          <rect
            data-hover-surface
            x={PAD.left}
            y={PAD.top}
            width={plotW}
            height={plotH}
            fill="transparent"
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
        {at !== null && (
          <div
            data-testid="readout"
            className="pointer-events-none absolute top-2 min-w-36 rounded-lg border border-border bg-surface/95 px-3 py-2 text-xs shadow-[var(--glow)]"
            style={x(at) > width / 2 ? { right: width - x(at) + 12 } : { left: x(at) + 12 }}
          >
            <p className="mb-1 font-semibold">{dayLabel(shownDays[at] ?? 0)}</p>
            {[...shown].reverse().map((s) => (
              <p key={s.id} className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
                <span className="text-muted">{s.label}</span>
                <span className="tabular ml-auto font-semibold">{format(s.values[at] ?? 0)}</span>
              </p>
            ))}
            {stacked && shown.length > 1 && (
              <p className="mt-1 flex justify-between border-t border-border pt-1">
                <span className="text-muted">Total</span>
                <span className="tabular font-semibold">{format(tops.at(-1)?.[at] ?? 0)}</span>
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
