import { formatMoney } from "../format.ts";

export type BarRow = {
  label: string;
  value: number;
  color?: string;
  testId?: string;
  /** Makes the row a key that opens its story. */
  onOpen?: (el: HTMLElement) => void;
};

/**
 * Ledger rows in a dark display (design §10, Money): each row its name, its amount in the dot
 * face and a lit level strip as long as its share of the largest; a row that opens its story is a
 * key (a button named "Open <name>").
 */
export function Bars({
  rows,
  format = formatMoney,
}: {
  rows: BarRow[];
  format?: (n: number) => string;
}) {
  const top = Math.max(1, ...rows.map((r) => Math.abs(r.value)));
  return (
    <ul className="display ledger">
      {rows.map((r) => (
        <li key={r.label} className={`ledger__row ${r.onOpen ? "ledger__row--key" : ""}`}>
          <span className="ledger__label">{r.label}</span>
          <span className="dotface dotface--xs ledger__amount" data-testid={r.testId}>
            {format(r.value)}
          </span>
          <span className="ledger__level" aria-hidden="true">
            <span
              className="bar-x"
              style={{
                width: `${(100 * Math.abs(r.value)) / top}%`,
                background: r.color ?? "var(--lamp)",
              }}
            />
          </span>
          {r.onOpen && (
            <button
              type="button"
              aria-label={`Open ${r.label}`}
              className="ledger__key"
              onClick={(e) => r.onOpen?.(e.currentTarget)}
            />
          )}
        </li>
      ))}
    </ul>
  );
}
