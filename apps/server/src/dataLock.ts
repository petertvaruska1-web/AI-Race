import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

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

/**
 * Claims the server's save folder for this process, so a second server can never
 * open the same database: two writers corrupt it. A lock left by a server that is
 * no longer running is taken over. Returns the function that lets the folder go.
 */
export function lockData(dir: string, pid = process.pid, isRunning = running): () => void {
  const file = join(dir, "server.lock");
  try {
    writeFileSync(file, `${pid}\n`, { flag: "wx" });
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "EEXIST") throw err;
    const holder = Number.parseInt(readFileSync(file, "utf8"), 10);
    if (Number.isInteger(holder) && holder !== pid && isRunning(holder)) {
      throw new Error(
        `Another game server (process ${holder}) is already using ${dir}. Stop it first, or give this one its own data folder.`,
      );
    }
    writeFileSync(file, `${pid}\n`);
  }
  return () => {
    try {
      if (Number.parseInt(readFileSync(file, "utf8"), 10) === pid) rmSync(file);
    } catch {
      // Already gone.
    }
  };
}
