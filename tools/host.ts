import { spawn, spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { networkInterfaces } from "node:os";
import { join, resolve } from "node:path";
import {
  ensureCatalog,
  healthy,
  type Proc,
  REPO,
  startServer,
  startWorker,
  stop,
  uvCommand,
  waitFor,
} from "./processes.ts";

// `npm run host`: play with friends from this PC (Stage 2, docs/hosting.md). It builds the web
// app, packs the helper for friends to download, starts the game server (serving the web app)
// and this PC's own graphics card, and, when cloudflared is installed, opens a free quick tunnel
// so friends can reach the game from anywhere. It never downloads anything itself.

const data = resolve(process.env.AIRACE_DATA ?? join(REPO, "data"));
const port = Number(process.env.AIRACE_SERVER_PORT ?? 8787);
const local = `http://127.0.0.1:${port}`;
// HOST=127.0.0.1 keeps the game to this PC (checks); by default friends on the network reach it.
const bind = process.env.HOST ?? "0.0.0.0";

/** Runs a command line (through the shell, so `npm` is found on Windows too). */
function run(what: string, command: string, cwd = REPO): void {
  console.log(what);
  const r = spawnSync(command, { cwd, stdio: "inherit", shell: true });
  if (r.status !== 0) {
    console.error(`That didn't work (${what.toLowerCase()}).`);
    process.exit(1);
  }
}

/** The helper's code for friends' PCs: `ml/src`, `pyproject.toml` and `uv.lock`, zipped. */
function packHelper(): void {
  mkdirSync(join(data, "helper"), { recursive: true });
  console.log("Packing the helper for your friends…");
  const [cmd, ...pre] = uvCommand();
  const pack = join(REPO, "tools", "helper", "pack.py");
  const zip = join(data, "helper", "ml.zip");
  const r = spawnSync(
    cmd as string,
    [...pre, "run", "--no-sync", "python", pack, join(REPO, "ml"), zip],
    { cwd: join(REPO, "ml"), stdio: "inherit" },
  );
  if (r.status !== 0) {
    console.error("That didn't work (packing the helper).");
    process.exit(1);
  }
}

/** Starts a free Cloudflare quick tunnel to the game and resolves to its https link. */
function openTunnel(): Promise<{ proc: Proc; url: string } | null> {
  // AIRACE_NO_TUNNEL=1: friends on this network only.
  if (process.env.AIRACE_NO_TUNNEL === "1") return Promise.resolve(null);
  const probe = spawnSync("cloudflared", ["--version"], { stdio: "ignore" });
  if (probe.status !== 0) return Promise.resolve(null);
  console.log("Opening a link your friends can reach (Cloudflare quick tunnel)…");
  const child = spawn("cloudflared", ["tunnel", "--url", `http://localhost:${port}`], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  const proc: Proc = { name: "tunnel", child, output: [] };
  return new Promise((done) => {
    const timer = setTimeout(() => done(null), 60_000);
    const take = (buf: Buffer) => {
      const text = buf.toString();
      proc.output.push(text);
      const m = /https:\/\/[a-z0-9-]+\.trycloudflare\.com/.exec(text);
      if (m) {
        clearTimeout(timer);
        done({ proc, url: m[0] });
      }
    };
    child.stdout?.on("data", take);
    child.stderr?.on("data", take);
    child.once("exit", () => done(null));
  });
}

function lanLinks(): string[] {
  if (bind !== "0.0.0.0") return [];
  return Object.values(networkInterfaces())
    .flat()
    .filter((a) => a && a.family === "IPv4" && !a.internal)
    .map((a) => `http://${a?.address}:${port}`);
}

run("Building the game's web app…", "npm run build -w apps/client");
if (!ensureCatalog(data)) process.exit(1);
packHelper();

const procs: Proc[] = [];
const shutdown = async () => {
  await Promise.all(procs.map(stop));
  process.exit(0);
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

// The tunnel comes first: the server must know the address friends use (setup files carry it).
const tunnel = await openTunnel();
if (tunnel) procs.push(tunnel.proc);
const publicUrl = tunnel?.url ?? lanLinks()[0] ?? `http://localhost:${port}`;

procs.push(
  startServer(
    {
      AIRACE_DATA: data,
      PORT: String(port),
      // Friends on the same network reach this PC directly; the tunnel reaches it locally.
      HOST: bind,
      AIRACE_CLIENT_DIST: join(REPO, "apps", "client", "dist"),
      AIRACE_PUBLIC_URL: publicUrl,
    },
    true,
  ),
);
await waitFor("the game server", healthy(local));
// This PC's own graphics card (the house card): it counts as the world host's.
procs.push(startWorker(["--server", `ws://127.0.0.1:${port}/worker`, "--data-root", data], true));

console.log("\nAI Race is running. Ctrl+C stops everything.\n");
console.log(`  You play at:      http://localhost:${port}`);
if (tunnel) {
  console.log(`  Friends play at:  ${tunnel.url}`);
  console.log("  (This link changes every time you host: send it to your friends each session.)");
} else {
  for (const link of lanLinks()) console.log(`  Friends on your network play at: ${link}`);
  console.log("\n  For friends elsewhere, install Cloudflare's free tunnel once, then host again:");
  console.log("    winget install Cloudflare.cloudflared");
}
console.log("");
