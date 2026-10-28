import type { ReactNode } from "react";
import { CountUp } from "./CountUp.tsx";

/** A big number with its label, rolling as it changes, and a green or red change below. */
export function Stat({
  label,
  value,
  delta,
  deltaLabel,
  format,
}: {
  label: ReactNode;
  value: number;
  delta?: number;
  /** What the change is measured over, e.g. "this week". */
  deltaLabel?: string;
  format: (n: number) => string;
}) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wider text-muted">{label}</p>
      <p className="font-display text-3xl font-bold">
        <CountUp value={value} format={format} />
      </p>
      {delta !== undefined && (
        <p
          className={`text-sm font-semibold ${delta > 0 ? "text-good" : delta < 0 ? "text-danger" : "text-muted"}`}
        >
          {delta > 0 ? "▲ " : delta < 0 ? "▼ " : ""}
          {delta === 0 ? "no change" : format(Math.abs(delta))}
          {deltaLabel && <span className="font-normal text-muted"> {deltaLabel}</span>}
        </p>
      )}
    </div>
  );
}
