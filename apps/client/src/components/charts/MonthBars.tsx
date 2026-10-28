import { monthName } from "../../format.ts";

export type MonthBar = { month: string; value: number };

/**
 * One bar a month, above the zero line or below it, growing in when shown.
 * With `onOpen` each month is a button that opens it.
 */
export function MonthBars({
  label,
  bars,
  format,
  onOpen,
  color = (v) => (v >= 0 ? "var(--good)" : "var(--danger)"),
  height = 160,
}: {
  label: string;
  bars: MonthBar[];
  format: (n: number) => string;
  onOpen?: (month: string, el: HTMLElement) => void;
  color?: (value: number) => string;
  height?: number;
}) {
  const top = Math.max(1, ...bars.map((b) => Math.abs(b.value)));
  const anyBelow = bars.some((b) => b.value < 0);
  const columns = bars.map((b) => {
    const size = `${(100 * Math.abs(b.value)) / top}%`;
    const name = `${monthName(b.month)}: ${format(b.value)}`;
    const body = (
      <>
        <span className="flex w-full flex-1 flex-col justify-end">
          {b.value > 0 && (
            <span
              className="bar-up block w-full rounded-t opacity-85"
              style={{ height: size, background: color(b.value) }}
            />
          )}
        </span>
        <span className="block h-px w-full bg-border" />
        {anyBelow && (
          <span className="flex w-full flex-1 flex-col">
            {b.value < 0 && (
              <span
                className="bar-down block w-full rounded-b opacity-85"
                style={{ height: size, background: color(b.value) }}
              />
            )}
          </span>
        )}
        <span className="mt-1 block text-[11px] text-muted">{monthName(b.month, true)}</span>
      </>
    );
    return onOpen ? (
      <button
        key={b.month}
        type="button"
        aria-label={name}
        title={name}
        onClick={(e) => onOpen(b.month, e.currentTarget)}
        className="lift-bar flex min-w-8 flex-1 flex-col items-center rounded-md px-0.5"
      >
        {body}
      </button>
    ) : (
      <span key={b.month} title={name} className="flex min-w-8 flex-1 flex-col items-center">
        {body}
      </span>
    );
  });
  const box = "flex items-stretch gap-2";
  return onOpen ? (
    <fieldset className={`${box} m-0 min-w-0 border-0 p-0`} style={{ height }} aria-label={label}>
      {columns}
    </fieldset>
  ) : (
    <div className={box} style={{ height }} role="img" aria-label={label}>
      {columns}
    </div>
  );
}
