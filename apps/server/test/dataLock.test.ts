import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { LOCK_STALE_MS, lockData } from "../src/dataLock.ts";

const folder = () => mkdtempSync(join(tmpdir(), "airace-lock-"));
const NOW = 1_000_000;
const at = () => NOW;
const lockOf = (dir: string) => readFileSync(join(dir, "server.lock"), "utf8").trim();

describe("the save-folder lock", () => {
  it("claims a free folder, saying who holds it and when, and lets it go", () => {
    const dir = folder();
    const release = lockData(dir, 111, () => false, at);
    expect(lockOf(dir)).toBe(`111 ${NOW}`);
    release();
    expect(existsSync(join(dir, "server.lock"))).toBe(false);
  });

  it("refuses a folder a running server holds, so two never share a database", () => {
    const dir = folder();
    writeFileSync(join(dir, "server.lock"), `222 ${NOW - 5_000}\n`);
    expect(() => lockData(dir, 111, (pid) => pid === 222, at)).toThrow(
      /Another game server \(process 222\) is already using/,
    );
    expect(lockOf(dir)).toBe(`222 ${NOW - 5_000}`);
  });

  it("takes over a lock left by a server that is no longer running", () => {
    const dir = folder();
    writeFileSync(join(dir, "server.lock"), `222 ${NOW}\n`);
    lockData(dir, 111, () => false, at);
    expect(lockOf(dir)).toBe(`111 ${NOW}`);
  });

  it("takes over a silent lock whose process id now belongs to another program", () => {
    const dir = folder();
    // The old server died; Windows gave its id to a browser, which is running.
    writeFileSync(join(dir, "server.lock"), `222 ${NOW - LOCK_STALE_MS - 1}\n`);
    lockData(dir, 111, () => true, at);
    expect(lockOf(dir)).toBe(`111 ${NOW}`);
    // A lock from before the heartbeat (only an id) is taken over too.
    writeFileSync(join(dir, "server.lock"), "333\n");
    lockData(dir, 111, () => true, at);
    expect(lockOf(dir)).toBe(`111 ${NOW}`);
  });
});
