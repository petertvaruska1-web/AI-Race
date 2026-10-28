import { formatMoney } from "../format.ts";

export type BarRow = {
  label: string;
  value: number;
  color?: string;
  testId?: string;
  /** Makes the row a button that opens its story. */
  onOpen?: (el: HTMLElement) => void;
};

/** Labelled bars, each as long as its share of the largest, growing in when shown. */
export function Bars({
  rows,
  format = formatMoney,
}: {
  rows: BarRow[];
  format?: (n: number) => string;
}) {
  const top = Math.max(1, ...rows.map((r) => Math.abs(r.value)));
  return (
    <ul className="space-y-2 text-sm">
      {rows.map((r) => (
        <li key={r.label} className={`relative ${r.onOpen ? "lift-row rounded-md" : ""}`}>
          <div className="flex justify-between gap-3">
            <span className="text-muted">{r.label}</span>
            <span className="tabular font-semibold" data-testid={r.testId}>
              {format(r.value)}
            </span>
          </div>
          <div className="mt-1 h-1.5 rounded-full bg-surface-2">
            <div
              className="bar-x h-1.5 rounded-full"
              style={{
                width: `${(100 * Math.abs(r.value)) / top}%`,
                background: r.color ?? "var(--brand)",
              }}
            />
          </div>
          {r.onOpen && (
            <button
              type="button"
              aria-label={`Open ${r.label}`}
              className="absolute inset-0 rounded-md"
              onClick={(e) => r.onOpen?.(e.currentTarget)}
            />
          )}
        </li>
      ))}
    </ul>
  );
}
