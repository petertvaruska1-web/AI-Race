// Game-pace reveal (spec §7.3): a worker may train faster than game time, but a
// player sees each training event only once game time reaches it, so faster
// hardware never shows anyone more, or sooner (R3).

export type RevealRun = { startDay: number; endDay: number; totalSteps: number };
export type StepEvent = { step: number };

/** How far through its game time a run is, counting the fraction of the current day. */
export function revealProgress(run: RevealRun, day: number, intraDayFraction: number): number {
  if (run.endDay <= run.startDay) return 1;
  const t = (day + intraDayFraction - run.startDay) / (run.endDay - run.startDay);
  return Math.min(1, Math.max(0, t));
}

/** How many of `events` (in step order) game time has reached. */
export function revealedCount(
  run: RevealRun,
  day: number,
  intraDayFraction: number,
  events: readonly StepEvent[],
): number {
  const limit = run.totalSteps * revealProgress(run, day, intraDayFraction);
  let n = 0;
  while (n < events.length && (events[n]?.step ?? Number.POSITIVE_INFINITY) <= limit) n++;
  return n;
}
