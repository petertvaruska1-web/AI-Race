import type { BalanceVersion } from "@airace/sim";
import type { Scheduler } from "./scheduler.ts";

export type { Scheduler } from "./scheduler.ts";

/** The deterministic sim's version; snapshots from another version are refused. */
export const SIM_VERSION = "sim-10";

export type ServerOptions = {
  /** PGlite data directory; null keeps the database in memory (tests). */
  dbDir: string | null;
  blobDir: string;
  catalogPath: string | null;
  /**
   * The current rival pack's folder (`data/rivals/v3`, built on the current training data): new
   * worlds draw their rivals from it. Null, missing or built on other data: new worlds run
   * without rivals.
   */
  rivalsDir?: string | null;
  /**
   * The earlier packs (`data/rivals/v2`, `data/rivals/v1`), whatever data they were built on:
   * they serve the rivals already seated from them, and worlds of the rules they were built
   * for keep drawing from them. Never built again.
   */
  earlierRivalsDirs?: readonly string[];
  /** Signs cookies, ws tokens and blob URLs. */
  secret: string;
  /** Shared secret ML workers present when they connect. */
  workerToken: string;
  balance: BalanceVersion;
  scheduler: Scheduler;
  /** Where players reach the server, e.g. http://localhost:8787. */
  publicUrl: string;
  logLevel?: string;
  /** The data root (`data/`): helpers download the game's data and code from it (Stage 2). */
  dataRoot?: string | null;
  /** The built web app (`apps/client/dist`), served at `/` when the host plays from their PC. */
  clientDist?: string | null;
};
