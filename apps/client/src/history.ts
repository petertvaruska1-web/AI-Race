import type { DaySampleView, PrivateCompany } from "@airace/protocol";

export type DayPoint = DaySampleView & { day: number };

/** The company's day history, oldest first (the view keys it by day). */
export function daysOf(company: PrivateCompany): DayPoint[] {
  return Object.entries(company.history)
    .map(([day, sample]) => ({ ...sample, day: Number(day) }))
    .sort((a, b) => a.day - b.day);
}
