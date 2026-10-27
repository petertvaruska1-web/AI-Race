import { createHmac, randomUUID, timingSafeEqual } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, rename, rm, stat } from "node:fs/promises";
import { dirname, resolve, sep } from "node:path";
import { type Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { FastifyInstance } from "fastify";
import type { AppContext } from "./app.ts";

/** Where checkpoints live. Workers reach it only through signed URLs; players never do (R2). */
export type BlobStore = {
  put(key: string, stream: Readable): Promise<void>;
  get(key: string): Promise<Readable | null>;
  exists(key: string): Promise<boolean>;
};

export type BlobMethod = "GET" | "PUT";

const KEY = /^[a-z0-9][a-z0-9/_.-]{0,200}$/;
export const MAX_BLOB_BYTES = 512 * 1024 * 1024;
const DEFAULT_TTL_MS = 3_600_000;

export class BlobKeyError extends Error {}

export function assertBlobKey(key: string): void {
  const segments = key.split("/");
  if (!KEY.test(key) || segments.some((s) => s === "" || s === "." || s === "..")) {
    throw new BlobKeyError(`invalid blob key: ${JSON.stringify(key)}`);
  }
}

export class LocalFsBlobStore implements BlobStore {
  private readonly root: string;

  constructor(dir: string) {
    this.root = resolve(dir);
  }

  private pathOf(key: string): string {
    assertBlobKey(key);
    const p = resolve(this.root, key);
    if (!p.startsWith(this.root + sep))
      throw new BlobKeyError(`blob key escapes the store: ${key}`);
    return p;
  }

  /** Writes to a temporary file and renames it into place, so a key is whole or absent. */
  async put(key: string, stream: Readable): Promise<void> {
    const p = this.pathOf(key);
    await mkdir(dirname(p), { recursive: true });
    const tmp = `${p}.${randomUUID()}.partial`;
    try {
      await pipeline(stream, createWriteStream(tmp));
      await rename(tmp, p);
    } catch (err) {
      await rm(tmp, { force: true });
      throw err;
    }
  }

  async get(key: string): Promise<Readable | null> {
    const p = this.pathOf(key);
    return (await this.exists(key)) ? createReadStream(p) : null;
  }

  async exists(key: string): Promise<boolean> {
    try {
      return (await stat(this.pathOf(key))).isFile();
    } catch (err) {
      if (err instanceof BlobKeyError) throw err;
      return false;
    }
  }
}

function blobSignature(ctx: AppContext, method: BlobMethod, key: string, exp: number): Buffer {
  return createHmac("sha256", ctx.opts.secret).update(`blob:${method} ${key} ${exp}`).digest();
}

/** A URL that allows `method` on `key` until it expires. */
export function signBlobUrl(
  ctx: AppContext,
  key: string,
  method: BlobMethod,
  ttlMs = DEFAULT_TTL_MS,
): string {
  assertBlobKey(key);
  const exp = ctx.opts.scheduler.now() + ttlMs;
  const sig = blobSignature(ctx, method, key, exp).toString("base64url");
  return `${ctx.opts.publicUrl}/blobs/${key}?exp=${exp}&sig=${sig}`;
}

function verifyBlobUrl(
  ctx: AppContext,
  method: BlobMethod,
  key: string,
  exp: string | undefined,
  sig: string | undefined,
): boolean {
  if (!exp || !sig || !/^\d+$/.test(exp)) return false;
  const expected = blobSignature(ctx, method, key, Number(exp));
  const given = Buffer.from(sig, "base64url");
  if (given.length !== expected.length || !timingSafeEqual(given, expected)) return false;
  return Number(exp) > ctx.opts.scheduler.now();
}

/** Counts bytes and fails the stream once it passes `limit`. */
function sizeLimit(limit: number): Transform {
  let seen = 0;
  return new Transform({
    transform(chunk: Buffer, _enc, done) {
      seen += chunk.length;
      if (seen > limit) done(Object.assign(new Error("blob too large"), { tooLarge: true }));
      else done(null, chunk);
    },
  });
}

export function registerBlobRoutes(app: FastifyInstance, ctx: AppContext): void {
  void app.register(async (scope) => {
    // Bodies stream straight to disk instead of being parsed.
    scope.removeAllContentTypeParsers();
    scope.addContentTypeParser("*", (_req, payload, done) => done(null, payload));

    type BlobRequest = { Params: { "*": string }; Querystring: { exp?: string; sig?: string } };

    const authorize = (method: BlobMethod, key: string, q: BlobRequest["Querystring"]) => {
      try {
        assertBlobKey(key);
      } catch {
        return false;
      }
      return verifyBlobUrl(ctx, method, key, q.exp, q.sig);
    };

    scope.get<BlobRequest>("/blobs/*", async (req, reply) => {
      const key = req.params["*"];
      if (!authorize("GET", key, req.query)) return reply.code(403).send({ error: "Forbidden." });
      const stream = await ctx.blobs.get(key);
      if (!stream) return reply.code(404).send({ error: "Not found." });
      return reply.type("application/octet-stream").send(stream);
    });

    scope.put<BlobRequest>("/blobs/*", async (req, reply) => {
      const key = req.params["*"];
      if (!authorize("PUT", key, req.query)) return reply.code(403).send({ error: "Forbidden." });
      try {
        // pipeline (unlike .pipe) passes a dropped connection on, so the store discards the file.
        const limited = sizeLimit(MAX_BLOB_BYTES);
        pipeline(req.raw, limited).catch(() => {});
        await ctx.blobs.put(key, limited);
      } catch (err) {
        if ((err as { tooLarge?: boolean }).tooLarge) {
          return reply.code(413).send({ error: "Too large." });
        }
        if (req.raw.destroyed || req.raw.aborted) return reply;
        throw err;
      }
      return reply.code(204).send();
    });
  });
}
