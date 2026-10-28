// Plain-language numbers for players (R7).

import { dateOfDay } from "@airace/sim";

export { formatMoney } from "@airace/sim";

const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

/** "2027-03" as "March 2027", or "Mar" when short. */
export function monthName(month: string, short = false): string {
  const name = MONTHS[Number(month.slice(5, 7)) - 1] ?? month;
  return short ? name.slice(0, 3) : `${name} ${month.slice(0, 4)}`;
}

/** A game day as its date, "Mar 3". */
export function dayLabel(day: number): string {
  const date = dateOfDay(day);
  return `${monthName(date.slice(0, 7), true)} ${Number(date.slice(8))}`;
}

/** A short number for chart axes: 950, 12k, 3.4M. */
export function formatCompact(n: number): string {
  const a = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (a >= 1e9) return `${sign}${trim(a / 1e9)}B`;
  if (a >= 1e6) return `${sign}${trim(a / 1e6)}M`;
  if (a >= 1e3) return `${sign}${trim(a / 1e3)}k`;
  return `${sign}${Math.round(a)}`;
}

function trim(x: number): string {
  return x >= 100 ? String(Math.round(x)) : x.toFixed(1).replace(/\.0$/, "");
}

/** A token count as words a player can picture ("about 32 million words"). */
export function words(tokens: number): string {
  if (tokens >= 1e6) {
    const m = tokens / 1e6;
    return `about ${m < 10 ? m.toFixed(1).replace(/\.0$/, "") : Math.round(m)} million words`;
  }
  if (tokens >= 1e3) return `about ${Math.round(tokens / 1e3)} thousand words`;
  return `${tokens} words`;
}

export function gameDays(n: number): string {
  return n === 1 ? "1 game day" : `${n} game days`;
}

/** A whole count with thousands separators ("12,480"), without the platform's locale rules. */
export function formatCount(n: number): string {
  const digits = String(Math.abs(Math.round(n)));
  return `${n < 0 ? "-" : ""}${digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}`;
}

export function percent(x: number): string {
  return `${Math.round(x * 100)}%`;
}

export function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** A change with its sign: "+12", "−$300", "0". */
export function signed(format: (n: number) => string): (n: number) => string {
  return (n) => `${n > 0 ? "+" : n < 0 ? "−" : ""}${format(Math.abs(n))}`;
}
