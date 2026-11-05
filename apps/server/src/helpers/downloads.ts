import { createHash } from "node:crypto";
import { createReadStream, existsSync, readFileSync } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import fastifyStatic from "@fastify/static";
import type { FastifyInstance, FastifyRequest } from "fastify";
import type { AppContext } from "../app.ts";
import { requireUser } from "../auth.ts";
import { helperKeyFor, userOfHelperKey } from "./keys.ts";

/** The helper's setup script and the template of its setup file. */
const HELPER_TOOLS = resolve(dirname(fileURLToPath(import.meta.url)), "../../../../tools/helper");
/** The data a helper needs, under the data root; nothing else there is ever served. */
export const HELPER_DATA_DIRS = ["corpus", "tokenizer", "judge", "novelty", "rivals"] as const;

export type ManifestFile = { path: string; size: number; sha256: string };
export type DataManifest = { dataIdentity: string | null; files: ManifestFile[] };

/** The player behind a request: a signed-in session, or a helper key as a bearer token. */
async function helperUser(ctx: AppContext, req: FastifyRequest): Promise<string | null> {
  const auth = req.headers.authorization;
  if (auth?.startsWith("Bearer ")) return userOfHelperKey(ctx, auth.slice("Bearer ".length).trim());
  try {
    return (await requireUser(ctx, req)).userId;
  } catch {
    return null;
  }
}

/**
 * What a friend's PC downloads from the host to become a helper (Stage 2): a setup file made for
 * them (`/helper/setup.cmd`, with the game's address and their key), its script, the helper's
 * code (`/helper/ml.zip`, packed by `npm run host`) and the game's data with a checksummed
 * manifest. Only for players: a session or a helper key. Server files are never reachable.
 */
export function registerHelperDownloads(app: FastifyInstance, ctx: AppContext): void {
  const root = ctx.opts.dataRoot ? resolve(ctx.opts.dataRoot) : null;
  const sums = new Map<string, { size: number; mtimeMs: number; sha256: string }>();

  app.register(async (scope) => {
    scope.addHook("onRequest", async (req, reply) => {
      if (!(await helperUser(ctx, req))) {
        return reply.code(401).send({ error: "Please sign in again." });
      }
    });

    scope.get("/helper/setup.cmd", async (req, reply) => {
      const userId = (await helperUser(ctx, req)) as string;
      const template = readFileSync(join(HELPER_TOOLS, "setup.cmd.template"), "utf8");
      const body = template
        .replaceAll("{{URL}}", ctx.opts.publicUrl.replace(/\/+$/, ""))
        .replaceAll("{{KEY}}", await helperKeyFor(ctx, userId))
        .replace(/\r?\n/g, "\r\n");
      return reply
        .header("content-type", "application/octet-stream")
        .header("content-disposition", 'attachment; filename="AI Race Helper setup.cmd"')
        .send(body);
    });

    scope.get("/helper/setup.ps1", async (_req, reply) =>
      reply
        .header("content-type", "text/plain; charset=utf-8")
        .send(readFileSync(join(HELPER_TOOLS, "setup.ps1"), "utf8")),
    );

    scope.get("/helper/ml.zip", async (_req, reply) => {
      const zip = root ? join(root, "helper", "ml.zip") : null;
      if (!zip || !existsSync(zip)) {
        return reply
          .code(404)
          .send({ error: "The helper isn't packed yet. Start the game with npm run host." });
      }
      return reply.header("content-type", "application/zip").send(createReadStream(zip));
    });

    scope.get("/helper/data/manifest.json", async (_req, reply) => {
      if (!root) return reply.code(404).send({ error: "This game has no data to share." });
      return manifest(root, ctx.catalog?.dataIdentity ?? null, sums);
    });

    if (root) {
      await scope.register(fastifyStatic, {
        root,
        prefix: "/helper/data/",
        decorateReply: false,
        index: false,
        list: false,
        dotfiles: "deny",
        allowedPath: (path) => HELPER_DATA_DIRS.some((d) => path.startsWith(`/${d}/`)),
      });
    }
  });
}

/** Every helper data file with its size and sha256; checksums are kept until a file changes. */
async function manifest(
  root: string,
  dataIdentity: string | null,
  sums: Map<string, { size: number; mtimeMs: number; sha256: string }>,
): Promise<DataManifest> {
  const files: ManifestFile[] = [];
  for (const dir of HELPER_DATA_DIRS) {
    for (const full of await walk(join(root, dir))) {
      const path = relative(root, full).split(sep).join("/");
      const st = await stat(full);
      let known = sums.get(path);
      if (!known || known.size !== st.size || known.mtimeMs !== st.mtimeMs) {
        known = { size: st.size, mtimeMs: st.mtimeMs, sha256: await sha256Of(full) };
        sums.set(path, known);
      }
      files.push({ path, size: known.size, sha256: known.sha256 });
    }
  }
  files.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  return { dataIdentity, files };
}

async function walk(dir: string): Promise<string[]> {
  let entries: import("node:fs").Dirent[];
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return [];
  }
  const out: string[] = [];
  for (const e of entries) {
    if (e.name.startsWith(".")) continue;
    const full = join(dir, e.name);
    if (e.isDirectory()) out.push(...(await walk(full)));
    else if (e.isFile()) out.push(full);
  }
  return out;
}

function sha256Of(path: string): Promise<string> {
  return new Promise((resolveHash, reject) => {
    const hash = createHash("sha256");
    createReadStream(path)
      .on("data", (chunk) => hash.update(chunk))
      .on("error", reject)
      .on("end", () => resolveHash(hash.digest("hex")));
  });
}
