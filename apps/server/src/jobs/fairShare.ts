/** Game days after which a company's card time counts half (Fair mode). */
export const HALF_LIFE_DAYS = 15;

/**
 * How much of a Fair-mode world's shared graphics cards each company has used lately: card
 * time in milliseconds, halving every 15 game days. The company with the least goes first.
 * Kept in memory: after a restart everyone starts even.
 */
export class FairShare {
  private used = new Map<string, { ms: number; day: number }>();

  private static key(worldId: string, companyId: string): string {
    return `${worldId}|${companyId}`;
  }

  /** Card time used, decayed to `day`. */
  usage(worldId: string, companyId: string, day: number): number {
    const u = this.used.get(FairShare.key(worldId, companyId));
    if (!u) return 0;
    return u.ms * 0.5 ** (Math.max(0, day - u.day) / HALF_LIFE_DAYS);
  }

  add(worldId: string, companyId: string, ms: number, day: number): void {
    const now = this.usage(worldId, companyId, day) + Math.max(0, ms);
    this.used.set(FairShare.key(worldId, companyId), { ms: now, day });
  }
}
