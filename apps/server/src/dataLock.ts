import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/** How often the holder says it is still there, and how long until a silent lock is stale. */
export const LOCK_BEAT_MS = 10_000;
export const LOCK_STALE_MS = 30_000;

/** Whether a process with this id is still running. */
function running(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    // EPERM: it runs, but belongs to someone else.
    return (err as NodeJS.ErrnoException).code === "EPERM";
  }
}

/** The lock's holder and when it last said it was there (null in a lock from before beats). */
function readLock(file: string): { pid: number; at: number | null } {
  const [pid, at] = readFileSync(file, "utf8").trim().split(/\s+/);
  const beat = Number(at);
  return { pid: Number.parseInt(pid ?? "", 10), at: Number.isFinite(beat) && at ? beat : null };
}

/**
 * Claims the server's save folder for this process, so a second server can never
 * open the same database: two writers corrupt it. The holder rewrites the lock every
 * 10 s. A lock is taken over when its process is gone, or when it has not been
 * rewritten for 30 s: Windows reuses process ids, so a dead server's id may now
 * belong to another program. Returns the function that lets the folder go.
 */
export function lockData(
  dir: string,
  pid = process.pid,
  isRunning = running,
  now: () => number = Date.now,
): () => void {
  const file = join(dir, "server.lock");
  const write = (flag?: "wx") => writeFileSync(file, `${pid} ${now()}\n`, flag ? { flag } : {});
  try {
    write("wx");
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "EEXIST") throw err;
    const holder = readLock(file);
    const fresh = holder.at !== null && now() - holder.at < LOCK_STALE_MS;
    if (Number.isInteger(holder.pid) && holder.pid !== pid && isRunning(holder.pid) && fresh) {
      throw new Error(
        `Another game server (process ${holder.pid}) is already using ${dir}. Stop it first, or give this one its own data folder.`,
      );
    }
    write();
  }
  // The real server keeps its claim fresh; tests that lock for another pid don't.
  const beat =
    pid === process.pid
      ? setInterval(() => {
          try {
            if (readLock(file).pid === pid) write();
          } catch {
            // The folder went away: nothing to keep.
          }
        }, LOCK_BEAT_MS)
      : null;
  beat?.unref();
  return () => {
    if (beat) clearInterval(beat);
    try {
      if (readLock(file).pid === pid) rmSync(file);
    } catch {
      // Already gone.
    }
  };
}
