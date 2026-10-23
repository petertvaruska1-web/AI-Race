import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// Spec §5.2: the sim is a pure reducer. Wall clocks, ambient randomness and
// host APIs would make replay diverge, so its source may not mention them.
const SRC = fileURLToPath(new URL("../src", import.meta.url));
const FORBIDDEN = /\bDate\b|Math\.random|performance\.|\bcrypto\.|from ['"]node:|import\(['"]node:/;

function sourceFiles(): string[] {
  return readdirSync(SRC, { recursive: true, encoding: "utf8" })
    .filter((f) => f.endsWith(".ts"))
    .map((f) => join(SRC, f));
}

describe("sim determinism rules", () => {
  it("scans the sim sources", () => {
    expect(sourceFiles().length).toBeGreaterThan(0);
  });

  it("uses no clock, ambient randomness or host module", () => {
    const offenders: string[] = [];
    for (const file of sourceFiles()) {
      readFileSync(file, "utf8")
        .split("\n")
        .forEach((line, i) => {
          if (FORBIDDEN.test(line)) offenders.push(`${file}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(offenders).toEqual([]);
  });
});
