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

const SERVER_PORT = 8797;
const CLIENT_PORT = 5183;

export type Stack = {
  url: string;
  /** Stops and starts the game server on the same data, as a crash and restart would. */
  restartServer(): Promise<void>;
  /** Waits until the worker has logged a line matching `pattern` (count: how many times). */
  workerLog(pattern: RegExp, count?: number): Promise<void>;
};

function uv(args: string[]): void {
  const [cmd, ...pre] = uvCommand();
  const r = spawnSync(cmd as string, [...pre, "run", ...UV_RUN, ...args], {
    cwd: join(REPO, "ml"),
    encoding: "utf8",
  });
  if (r.status !== 0) throw new Error(`${args.join(" ")} failed:\n${r.stdout}\n${r.stderr}`);
}

/**
 * A whole game on this computer: a tiny corpus built from the test fixtures, the server on
 * the fast test balance, a real CPU worker and the web app.
 */
export const test = base.extend<{ stack: Stack }>({
  // biome-ignore lint/correctness/noEmptyPattern: Playwright fixtures take their dependencies this way
  stack: async ({}, use) => {
    const tmp = mkdtempSync(join(tmpdir(), "airace-e2e-"));
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
    // Tiny rivals (Phase 3): the server seats them in the new world.
    uv(["airace-ml", "build-rivals", "--data-root", data, "--quick", "--device", "cpu"]);
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
    const worker = startWorker(
      ["--server", `ws://127.0.0.1:${SERVER_PORT}/worker`, "--data-root", data, "--device", "cpu"],
      echo,
    );
    const client = startVite(CLIENT_PORT, { AIRACE_SERVER: serverUrl }, echo);
    const url = `http://127.0.0.1:${CLIENT_PORT}`;
    await waitFor("the web app", async () => (await fetch(url)).ok);
    await waitFor("the worker", () => worker.output.some((l) => /connected to/.test(l)), 120_000);

    await use({
      url,
      async restartServer() {
        await stop(server);
        server = startServer(env, echo);
        await waitFor("the restarted game server", healthy(serverUrl));
      },
      async workerLog(pattern, count = 1) {
        await waitFor(
          `the worker to log ${pattern}`,
          () => worker.output.filter((l) => pattern.test(l)).length >= count,
          300_000,
        );
      },
    });

    await Promise.all([stop(client), stop(worker), stop(server)]);
    rmSync(tmp, { recursive: true, force: true });
  },
});

export { expect } from "@playwright/test";
