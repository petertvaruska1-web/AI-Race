/** Time for everything the server schedules, so tests can run hours in milliseconds. */
export type TimerHandle = { readonly id: number };

export type Scheduler = {
  now(): number;
  setTimeout(fn: () => unknown, ms: number): TimerHandle;
  clearTimeout(h: TimerHandle): void;
};

export const realScheduler: Scheduler = {
  now: () => Date.now(),
  setTimeout(fn, ms) {
    const t = globalThis.setTimeout(() => void fn(), ms);
    return { id: Number(t[Symbol.toPrimitive]()) };
  },
  clearTimeout(h) {
    globalThis.clearTimeout(h.id);
  },
};

export type FakeScheduler = Scheduler & {
  /** Runs every timer due within `ms`, in time order, awaiting each one. */
  advance(ms: number): Promise<void>;
  pending(): number;
};

export function fakeScheduler(start = 1_000_000): FakeScheduler {
  let now = start;
  let nextId = 1;
  const timers = new Map<number, { due: number; fn: () => unknown }>();

  function earliest(limit: number): [number, { due: number; fn: () => unknown }] | null {
    let best: [number, { due: number; fn: () => unknown }] | null = null;
    for (const entry of timers) {
      if (entry[1].due > limit) continue;
      if (
        !best ||
        entry[1].due < best[1].due ||
        (entry[1].due === best[1].due && entry[0] < best[0])
      ) {
        best = entry;
      }
    }
    return best;
  }

  return {
    now: () => now,
    setTimeout(fn, ms) {
      const id = nextId++;
      timers.set(id, { due: now + Math.max(0, ms), fn });
      return { id };
    },
    clearTimeout(h) {
      timers.delete(h.id);
    },
    async advance(ms) {
      const target = now + ms;
      for (;;) {
        const next = earliest(target);
        if (!next) break;
        timers.delete(next[0]);
        now = next[1].due;
        await next[1].fn();
        await flush();
      }
      now = target;
      await flush();
    },
    pending: () => timers.size,
  };
}

/** Lets queued promise callbacks and I/O callbacks run. */
export function flush(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}
