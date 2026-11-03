import type { Economy } from "./balance.ts";

// Office and support (playtest pack §5, C35): a bigger company pays bigger bills, a
// growing share of what it earns, so money stays meaningful as it scales.

/**
 * The office and support bill for a day that earned `dayRevenue`: base × R^exponent a
 * month for a monthly revenue R, so the share of revenue rises slowly with size.
 */
export function overheadForDay(e: Economy, dayRevenue: number): number {
  if (dayRevenue <= 0) return 0;
  return (e.overhead.base * (dayRevenue * 30) ** e.overhead.exponent) / 30;
}
