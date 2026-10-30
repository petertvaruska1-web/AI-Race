import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { type Catalog, RivalPack } from "@airace/protocol";
import type { Logger } from "../log.ts";

export type LoadedPack = { pack: RivalPack; dir: string };

/**
 * Reads `rivals.json` from the rival pack (`airace-ml build-rivals`, game-layer
 * design §9). A missing pack, or one built from other training data, is a
 * warning: worlds then run without rivals. A file that doesn't parse is an
 * error, so a bad build is noticed.
 */
export function loadRivalPack(
  dir: string | null,
  catalog: Catalog | null,
  log: Logger,
): LoadedPack | null {
  const file = dir ? join(dir, "rivals.json") : null;
  if (!dir || !file || !existsSync(file)) {
    log.warn({ dir }, "no rival pack; worlds run without rivals until it is built");
    return null;
  }
  const pack = RivalPack.parse(JSON.parse(readFileSync(file, "utf8")));
  if (!catalog || pack.dataIdentity !== catalog.dataIdentity) {
    log.warn(
      { dir, pack: pack.dataIdentity, catalog: catalog?.dataIdentity ?? null },
      "the rival pack was built from other training data; build it again",
    );
    return null;
  }
  if (pack.quick) log.warn({ dir }, "the rival pack is a quick build: tiny rivals for checks");
  return { pack, dir };
}
