// How often the glass and the 3D objects draw (design §11, R3): the player's own graphics
// card trains their AI while they play, so nothing draws more than it must.

export type FramePolicy = { fps: number; still: boolean };

/**
 * Nothing while the tab is hidden; one still frame per change for reduced motion and lighter
 * graphics; 60 frames a second while the player drags something; 30 otherwise.
 */
export function framePolicy(o: {
  reducedMotion: boolean;
  lighter: boolean;
  hidden: boolean;
  interacting: boolean;
}): FramePolicy {
  if (o.hidden || o.reducedMotion || o.lighter) return { fps: 0, still: true };
  return { fps: o.interacting ? 60 : 30, still: false };
}

/** Whether a frame at `fps` is due at `nowMs`, the last one drawn at `lastMs`. */
export function due(nowMs: number, lastMs: number, fps: number): boolean {
  return fps > 0 && nowMs - lastMs >= 1000 / fps - 1;
}

const SAMPLES = 90;
const SLOW_MS = 40;

/**
 * Watches frame times; `record` answers true once, the first time the last 90 frames' median
 * is above 40 ms (the game then switches on lighter graphics by itself).
 */
export function createFrameMeter(): { record(frameMs: number): boolean } {
  const times: number[] = [];
  let told = false;
  return {
    record(frameMs: number): boolean {
      times.push(frameMs);
      if (times.length > SAMPLES) times.shift();
      if (told || times.length < SAMPLES) return false;
      const sorted = [...times].sort((a, b) => a - b);
      const median = ((sorted[SAMPLES / 2 - 1] ?? 0) + (sorted[SAMPLES / 2] ?? 0)) / 2;
      if (median <= SLOW_MS) return false;
      told = true;
      return true;
    },
  };
}
