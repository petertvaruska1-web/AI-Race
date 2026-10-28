import type { ReactNode } from "react";
import { CountUp } from "./CountUp.tsx";

/** A big number with its label, rolling as it changes, and a green or red change below. */
export function Stat({
  label,
  value,
  delta,
  deltaLabel,
  format,
  testId,
  downIsGood = false,
}: {
  label: ReactNode;
  value: number;
  delta?: number;
  /** What the change is measured over, e.g. "this week". */
  deltaLabel?: string;
  format: (n: number) => string;
  testId?: string;
  /** For costs: a fall is good news (green) and a rise bad (red). */
  downIsGood?: boolean;
}) {
  const good = downIsGood ? (delta ?? 0) < 0 : (delta ?? 0) > 0;
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wider text-muted">{label}</p>
      <p className="font-display text-3xl font-bold" data-testid={testId}>
        <CountUp value={value} format={format} />
      </p>
      {delta !== undefined && (
        <p
          className={`text-sm font-semibold ${delta === 0 ? "text-muted" : good ? "text-good" : "text-danger"}`}
        >
          {delta > 0 ? "▲ " : delta < 0 ? "▼ " : ""}
          {delta === 0 ? "no change" : format(Math.abs(delta))}
          {deltaLabel && <span className="font-normal text-muted"> {deltaLabel}</span>}
        </p>
      )}
    </div>
  );
}
