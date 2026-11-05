import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test as base } from "@playwright/test";
import {
  healthy,
  type Proc,
  REPO,
  startServer,
  startVite,
  startWorker,
  stop,
  UV_RUN,
  uvCommand,
  waitFor,
} from "../../tools/processes.ts";

const SERVER_PORT = 8798;
const CLIENT_PORT = 5184;

/** A helper (an ML worker) on a "player's PC": here, another CPU worker on this computer. */
export type Helper = {
  proc: Proc;
  /** Waits until this helper has logged a line matching `pattern` `count` times. */
  log(pattern: RegExp, count?: number): Promise<void>;
  /** How many lines matching `pattern` it has logged. */
  count(pattern: RegExp): number;
  stop(): Promise<void>;
};

export type FriendsStack = {
  url: string;
  /** The PC running the game: the house card, the world host's. */
  house: Helper;
  /** Connects a helper with a player's own key. */
  startHelper(key: string, id: string): Promise<Helper>;
  restartServer(): Promise<void>;
};

function uv(args: string[]): void {
  const [cmd, ...pre] = uvCommand();
  const r = spawnSync(cmd as string, [...pre, "run", ...UV_RUN, ...args], {
    cwd: join(REPO, "ml"),
    encoding: "utf8",
  });
  if (r.status !== 0) throw new Error(`${args.join(" ")} failed:\n${r.stdout}\n${r.stderr}`);
}

function helperOf(proc: Proc): Helper {
  const count = (pattern: RegExp) => proc.output.filter((l) => pattern.test(l)).length;
  return {
    proc,
    count,
    log: (pattern, n = 1) =>
      waitFor(`a helper to log ${pattern}`, () => count(pattern) >= n, 300_000),
    stop: () => stop(proc),
  };
}

/**
 * Stage 2 end to end: a tiny corpus, the server on the fast test balance, the web app, the house
 * card, and helpers started with players' own keys (all CPU workers on this computer). Built once
 * for the file: each test plays its own world.
 */
export const test = base.extend<object, { friends: FriendsStack }>({
  friends: [
    // biome-ignore lint/correctness/noEmptyPattern: Playwright fixtures take their dependencies this way
    async ({}, use) => {
      const tmp = mkdtempSync(join(tmpdir(), "airace-e2e-friends-"));
      const data = join(tmp, "data");
      uv([
        "airace-content",
        "build",
        "--scale",
        "tiny",
        "--fixtures",
        "tests/fixtures/sources",
        "--out",
        data,
      ]);
      uv(["airace-ml", "export-catalog", "--data-root", data]);
      const serverUrl = `http://127.0.0.1:${SERVER_PORT}`;
      const env = {
        AIRACE_DATA: data,
        AIRACE_BALANCE: "balance-test",
        PORT: String(SERVER_PORT),
        LOG_LEVEL: "warn",
      };
      const echo = process.env.E2E_ECHO === "1";
      let server: Proc = startServer(env, echo);
      await waitFor("the game server", healthy(serverUrl));
      const worker = (args: string[]) =>
        startWorker(
          [
            "--server",
            `ws://127.0.0.1:${SERVER_PORT}/worker`,
            "--data-root",
            data,
            "--device",
            "cpu",
            ...args,
          ],
          echo,
        );
      const house = helperOf(worker(["--id", "house"]));
      const client = startVite(CLIENT_PORT, { AIRACE_SERVER: serverUrl }, echo);
      const url = `http://127.0.0.1:${CLIENT_PORT}`;
      await waitFor("the web app", async () => (await fetch(url)).ok);
      await house.log(/connected to/);
      const helpers: Helper[] = [house];

      await use({
        url,
        house,
        async startHelper(key, id) {
          const h = helperOf(worker(["--token", key, "--id", id]));
          helpers.push(h);
          await h.log(/connected to/);
          return h;
        },
        async restartServer() {
          await stop(server);
          server = startServer(env, echo);
          await waitFor("the restarted game server", healthy(serverUrl));
        },
      });

      await Promise.all([stop(client), ...helpers.map((h) => h.stop()), stop(server)]);
      rmSync(tmp, { recursive: true, force: true });
    },
    { scope: "worker", timeout: 600_000 },
  ],
});

export { expect } from "@playwright/test";
