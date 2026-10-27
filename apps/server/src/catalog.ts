import { existsSync, readFileSync } from "node:fs";
import { Catalog, DATASET_IDS, type PublicCatalog } from "@airace/protocol";
import type { Logger } from "./log.ts";

export const NO_TRAINING_DATA = "The training data is not installed yet.";

/**
 * Reads `catalog.json` (exported by `ml/`, plan Task 16). A missing file is a
 * warning: the game runs, but training is refused until the data is installed.
 * A file that exists but doesn't parse is an error, so a bad export is noticed.
 */
export function loadCatalog(path: string | null, log: Logger): Catalog | null {
  if (!path || !existsSync(path)) {
    log.warn({ path }, "no dataset catalog; training is disabled until the data is installed");
    return null;
  }
  const catalog = Catalog.parse(JSON.parse(readFileSync(path, "utf8")));
  catalog.datasets.sort((a, b) => DATASET_IDS.indexOf(a.id) - DATASET_IDS.indexOf(b.id));
  return catalog;
}

/** The catalog players may read: everything but the reference models' trait data. */
export function publicCatalog(c: Catalog): PublicCatalog {
  return {
    version: c.version,
    dataIdentity: c.dataIdentity,
    tokenizer: c.tokenizer,
    suite: c.suite,
    datasets: c.datasets,
  };
}
