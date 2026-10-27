// Plain-language numbers for players (R7).

export { formatMoney } from "@airace/sim";

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

export function percent(x: number): string {
  return `${Math.round(x * 100)}%`;
}

export function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
