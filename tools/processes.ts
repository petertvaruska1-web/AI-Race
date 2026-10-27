import { type ChildProcess, spawn, spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Starting and stopping the game's processes, shared by `npm run dev` and the end-to-end tests.

export const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);

/** How to run uv here: `uv` when it is on PATH, else `python -m uv` (the owner's PC; CLAUDE.md). */
export function uvCommand(): string[] {
  const probe = spawnSync("uv", ["--version"], { stdio: "ignore", shell: false });
  return probe.status === 0
    ? ["uv"]
    : [process.platform === "win32" ? "python" : "python3", "-m", "uv"];
}

/**
 * `uv run` options: a plain `uv run` re-syncs the environment without the content extra and
 * removes the `datasets` package (CLAUDE.md), so every run keeps it.
 */
export const UV_RUN = ["--extra", "content"];

export type Proc = { name: string; child: ChildProcess; output: string[] };

export function start(
  name: string,
  cmd: string,
  args: string[],
  opts: { cwd?: string; env?: Record<string, string>; echo?: boolean } = {},
): Proc {
  const child = spawn(cmd, args, {
    cwd: opts.cwd ?? REPO,
    env: { ...process.env, ...opts.env },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const proc: Proc = { name, child, output: [] };
  const take = (buf: Buffer) => {
    for (const line of buf.toString().split(/\r?\n/)) {
      if (!line) continue;
      proc.output.push(line);
      if (opts.echo) console.log(`[${name}] ${line}`);
    }
  };
  child.stdout?.on("data", take);
  child.stderr?.on("data", take);
  return proc;
}

/** The game server under Node, without a wrapper process that would outlive it. */
export function startServer(env: Record<string, string>, echo = false): Proc {
  const tsx = join(dirname(require.resolve("tsx/package.json")), "dist", "cli.mjs");
  return start("server", process.execPath, [tsx, "apps/server/src/main.ts"], { env, echo });
}

export function startWorker(args: string[], echo = false): Proc {
  const [cmd, ...pre] = uvCommand();
  return start(
    "worker",
    cmd as string,
    [...pre, "run", ...UV_RUN, "airace-ml", "worker", ...args],
    {
      cwd: join(REPO, "ml"),
      echo,
    },
  );
}

export function startVite(port: number, env: Record<string, string>, echo = false): Proc {
  const vite = join(
    dirname(require.resolve("vite/package.json", { paths: [join(REPO, "apps/client")] })),
    "bin",
    "vite.js",
  );
  return start(
    "client",
    process.execPath,
    [vite, "--port", String(port), "--strictPort", "--host", "127.0.0.1"],
    {
      cwd: join(REPO, "apps/client"),
      env,
      echo,
    },
  );
}

/** Stops a process and everything it started (uv starts Python as a child). */
export function stop(proc: Proc): Promise<void> {
  return new Promise((done) => {
    const { child } = proc;
    if (child.exitCode !== null || child.pid === undefined) return done();
    child.once("exit", () => done());
    if (process.platform === "win32")
      spawnSync("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore" });
    else child.kill("SIGTERM");
  });
}

export async function waitFor(
  what: string,
  check: () => Promise<boolean> | boolean,
  timeoutMs = 60_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      if (await check()) return;
    } catch {
      // not ready yet
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`timed out waiting for ${what}`);
}

export const healthy = (url: string) => async () => (await fetch(`${url}/api/health`)).ok;
