import type { ReactNode } from "react";
import { CountUp } from "./CountUp.tsx";

/** A big number in a display under its engraved label, rolling as it changes, and a green or red change below. */
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
      <p className="engrave">{label}</p>
      <div className="display mt-1.5 inline-block min-w-[9ch] px-3 py-2">
        <p className="dot dot--lg m-0" data-testid={testId}>
          <CountUp value={value} format={format} />
        </p>
      </div>
      {delta !== undefined && (
        <p
          className={`mt-1.5 text-sm font-semibold ${delta === 0 ? "text-muted" : good ? "text-good" : "text-danger"}`}
        >
          {delta > 0 ? "▲ " : delta < 0 ? "▼ " : ""}
          {delta === 0 ? "no change" : format(Math.abs(delta))}
          {deltaLabel && <span className="font-normal text-muted"> {deltaLabel}</span>}
        </p>
      )}
    </div>
  );
}
