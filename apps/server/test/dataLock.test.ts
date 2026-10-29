import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { lockData } from "../src/dataLock.ts";

const folder = () => mkdtempSync(join(tmpdir(), "airace-lock-"));

describe("the save-folder lock", () => {
  it("claims a free folder and lets it go", () => {
    const dir = folder();
    const release = lockData(dir, 111, () => false);
    expect(readFileSync(join(dir, "server.lock"), "utf8").trim()).toBe("111");
    release();
    expect(existsSync(join(dir, "server.lock"))).toBe(false);
  });

  it("refuses a folder a running server holds, so two never share a database", () => {
    const dir = folder();
    writeFileSync(join(dir, "server.lock"), "222\n");
    expect(() => lockData(dir, 111, (pid) => pid === 222)).toThrow(
      /Another game server \(process 222\) is already using/,
    );
    expect(readFileSync(join(dir, "server.lock"), "utf8").trim()).toBe("222");
  });

  it("takes over a lock left by a server that is no longer running", () => {
    const dir = folder();
    writeFileSync(join(dir, "server.lock"), "222\n");
    lockData(dir, 111, () => false);
    expect(readFileSync(join(dir, "server.lock"), "utf8").trim()).toBe("111");
  });
});
