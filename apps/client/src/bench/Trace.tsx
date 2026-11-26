import { useId } from "react";

/**
 * A line of values on an oscilloscope display (design §10): the trace in the lamp color with a
 * soft glow over a faint grid, and a dashed zero line when values cross it. Draws itself in.
 */
export function Trace({
  values,
  label,
  zeroLine = false,
  className = "",
}: {
  values: number[];
  label: string;
  zeroLine?: boolean;
  className?: string;
}) {
  const gradient = `trace${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const w = 280;
  const h = 64;
  const lo = Math.min(0, ...values);
  const hi = Math.max(0, ...values);
  const y = (v: number) => (hi === lo ? h / 2 : h - 4 - ((v - lo) / (hi - lo)) * (h - 8));
  const x = (i: number) => (values.length < 2 ? w / 2 : (i / (values.length - 1)) * w);
  const points = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={label}
      className={`block h-16 w-full ${className}`}
    >
      <defs>
        <linearGradient id={gradient} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="var(--lamp)" stopOpacity={0.32} />
          <stop offset="1" stopColor="var(--lamp)" stopOpacity={0} />
        </linearGradient>
      </defs>
      {[0.25, 0.5, 0.75].map((k) => (
        <line
          key={k}
          x1={0}
          x2={w}
          y1={h * k}
          y2={h * k}
          stroke="currentColor"
          strokeOpacity={0.08}
          vectorEffect="non-scaling-stroke"
        />
      ))}
      {zeroLine && (
        <line
          x1={0}
          x2={w}
          y1={y(0)}
          y2={y(0)}
          stroke="currentColor"
          strokeOpacity={0.3}
          strokeDasharray="3 4"
          vectorEffect="non-scaling-stroke"
        />
      )}
      {values.length > 1 && (
        <polygon points={`0,${h} ${points} ${w},${h}`} fill={`url(#${gradient})`} />
      )}
      <polyline
        points={points}
        fill="none"
        stroke="var(--lamp)"
        strokeWidth={2}
        strokeLinejoin="round"
        pathLength={1}
        vectorEffect="non-scaling-stroke"
        className="chart-draw"
        style={{ filter: "drop-shadow(0 0 4px var(--lamp))" }}
      />
    </svg>
  );
}
