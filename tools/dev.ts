import { join, resolve } from "node:path";
import {
  ensureCatalog,
  healthy,
  type Proc,
  REPO,
  startServer,
  startVite,
  startWorker,
  stop,
  waitFor,
} from "./processes.ts";

// `npm run dev`: the game server, an ML worker on this computer and the web app, together.

const data = resolve(process.env.AIRACE_DATA ?? join(REPO, "data"));
// Not PORT: tools that launch `npm run dev` often set PORT to the web app's port.
const port = Number(process.env.AIRACE_SERVER_PORT ?? 8787);
const server = `http://127.0.0.1:${port}`;
// A second game can run beside the first, with its own ports and data root.
const webPort = Number(process.env.AIRACE_WEB_PORT ?? 5173);

if (!ensureCatalog(data)) process.exit(1);

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
procs.push(startVite(webPort, { AIRACE_SERVER: server }, true));
console.log(`\nAI Race is running at http://localhost:${webPort} (Ctrl+C stops everything)\n`);
