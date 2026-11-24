import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { type Catalog, RivalPack } from "@airace/protocol";
import type { Logger } from "../log.ts";

export type LoadedPack = { pack: RivalPack; dir: string; keys: ReadonlySet<string> };

/**
 * Reads `rivals.json` from a rival pack (`airace-ml build-rivals`, game-layer design §9). A
 * file that doesn't parse is an error, so a bad build is noticed.
 *
 * The current pack (`current: true`) must be built on the catalog's training data: missing or
 * built on other data, it is a warning and new worlds run without rivals. An earlier pack is
 * loaded whatever data it was built on, because the rivals seated from it keep their AIs; a
 * missing one is not a warning (a new install has none).
 */
export function loadRivalPack(
  dir: string | null,
  catalog: Catalog | null,
  log: Logger,
  { current }: { current: boolean } = { current: true },
): LoadedPack | null {
  const file = dir ? join(dir, "rivals.json") : null;
  if (!dir || !file || !existsSync(file)) {
    if (current)
      log.warn({ dir }, "no rival pack; new worlds run without rivals until it is built");
    return null;
  }
  const pack = RivalPack.parse(JSON.parse(readFileSync(file, "utf8")));
  if (current && (!catalog || pack.dataIdentity !== catalog.dataIdentity)) {
    log.warn(
      { dir, pack: pack.dataIdentity, catalog: catalog?.dataIdentity ?? null },
      "the rival pack was built from other training data; build it again",
    );
    return null;
  }
  if (!current) log.info({ dir, version: pack.version }, "an earlier rival pack serves its rivals");
  if (pack.quick) log.warn({ dir }, "the rival pack is a quick build: tiny rivals for checks");
  const keys = new Set(pack.lineages.flatMap((l) => l.versions.map((v) => v.key)));
  return { pack, dir, keys };
}
