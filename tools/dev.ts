import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
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
} from "./processes.ts";

// `npm run dev`: the game server, an ML worker on this computer and the web app, together.

const data = resolve(process.env.AIRACE_DATA ?? join(REPO, "data"));
// Not PORT: tools that launch `npm run dev` often set PORT to the web app's port.
const port = Number(process.env.AIRACE_SERVER_PORT ?? 8787);
const server = `http://127.0.0.1:${port}`;

if (!existsSync(join(data, "catalog", "v1", "catalog.json"))) {
  console.log("Writing the dataset catalog for the game server…");
  const [cmd, ...pre] = uvCommand();
  const population = join(REPO, "ml", "runs", "gate-1", "runs");
  const args = [...pre, "run", ...UV_RUN, "airace-ml", "export-catalog", "--data-root", data];
  if (existsSync(population)) args.push("--population", population);
  const r = spawnSync(cmd as string, args, { cwd: join(REPO, "ml"), stdio: "inherit" });
  if (r.status !== 0) {
    console.error("Could not write the catalog. Build the training data first (see CLAUDE.md).");
    process.exit(1);
  }
}

const procs: Proc[] = [];
const shutdown = async () => {
  await Promise.all(procs.map(stop));
  process.exit(0);
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

procs.push(startServer({ AIRACE_DATA: data, PORT: String(port) }, true));
await waitFor("the game server", healthy(server));
// The server wrote the worker token on its first start; the worker reads it from the data root.
procs.push(startWorker(["--server", `ws://127.0.0.1:${port}/worker`, "--data-root", data], true));
procs.push(startVite(5173, { AIRACE_SERVER: server }, true));
console.log("\nAI Race is running at http://localhost:5173 (Ctrl+C stops everything)\n");
