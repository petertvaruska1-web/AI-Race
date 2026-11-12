import { createHmac, randomUUID, timingSafeEqual } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, rename, rm, stat } from "node:fs/promises";
import { dirname, resolve, sep } from "node:path";
import { type Readable, Transform, Writable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { FastifyInstance } from "fastify";
import type { AppContext } from "./app.ts";

/** Where checkpoints live. Workers reach it only through signed URLs; players never do (R2). */
export type BlobStore = {
  put(key: string, stream: Readable): Promise<void>;
  get(key: string): Promise<Readable | null>;
  exists(key: string): Promise<boolean>;
  /** Removes `key`; nothing happens when it is absent. */
  delete(key: string): Promise<void>;
  /**
   * One part (`index` of `total`, from 0) of a blob sent in pieces, in order. The key is whole
   * or absent as with `put`: it appears when the last part is in. A part that comes again right
   * after it was stored is kept once (its answer was lost on the way); part 0 always starts the
   * blob over; any other part out of turn is "out-of-turn" and the sender starts over.
   */
  putPart(key: string, index: number, total: number, stream: Readable): Promise<PartResult>;
};

export type PartResult = "stored" | "complete" | "out-of-turn";

export type BlobMethod = "GET" | "PUT";

const KEY = /^[a-z0-9][a-z0-9/_.-]{0,200}$/;
/**
 * The most a blob may hold: the saved progress of the largest brain (40.7M parameters with its
 * optimizer and roll-back copy) is about 650 MB.
 */
export const MAX_BLOB_BYTES = 1024 * 1024 * 1024;
/**
 * Big files come in parts: the free tunnel friends play through refuses any single upload over
 * 100 MB (the helper sends 32 MB parts; `ml/src/airace_ml/worker/blobs.py`).
 */
export const MAX_PART_BYTES = 64 * 1024 * 1024;
export const MAX_PARTS = 64;
const DEFAULT_TTL_MS = 3_600_000;

export class BlobKeyError extends Error {}

export function assertBlobKey(key: string): void {
  const segments = key.split("/");
  if (!KEY.test(key) || segments.some((s) => s === "" || s === "." || s === "..")) {
    throw new BlobKeyError(`invalid blob key: ${JSON.stringify(key)}`);
  }
}

/** Reads a stream to its end and keeps nothing. */
const drain = (stream: Readable) =>
  pipeline(stream, new Writable({ write: (_chunk, _enc, done) => done() }));

const tooLarge = () => Object.assign(new Error("blob too large"), { tooLarge: true });

export class LocalFsBlobStore implements BlobStore {
  private readonly root: string;
  /** Blobs arriving in parts: the part each expects next, of how many, and the bytes so far. */
  private readonly arriving = new Map<string, { next: number; total: number; bytes: number }>();

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

  async putPart(key: string, index: number, total: number, stream: Readable): Promise<PartResult> {
    const p = this.pathOf(key);
    const staging = `${p}.parts.partial`;
    let state = this.arriving.get(key);
    if (index === 0) state = { next: 0, total, bytes: 0 };
    if (state && state.total === total && index > 0 && index === state.next - 1) {
      await drain(stream); // stored already: only its answer was lost
      return "stored";
    }
    if (!state || state.total !== total || index !== state.next) {
      await drain(stream);
      return "out-of-turn";
    }
    await mkdir(dirname(p), { recursive: true });
    const tmp = `${p}.${randomUUID()}.partial`;
    try {
      await pipeline(stream, createWriteStream(tmp));
      const size = (await stat(tmp)).size;
      if (state.bytes + size > MAX_BLOB_BYTES) throw tooLarge();
      // A part joins the others only when all of it arrived, so one cut short is never kept.
      if (index === 0) await rename(tmp, staging);
      else await pipeline(createReadStream(tmp), createWriteStream(staging, { flags: "a" }));
      state.bytes += size;
      state.next += 1;
    } catch (err) {
      this.arriving.delete(key);
      throw err;
    } finally {
      await rm(tmp, { force: true });
    }
    if (state.next < total) {
      this.arriving.set(key, state);
      return "stored";
    }
    this.arriving.delete(key);
    await rename(staging, p);
    return "complete";
  }

  async get(key: string): Promise<Readable | null> {
    const p = this.pathOf(key);
    return (await this.exists(key)) ? createReadStream(p) : null;
  }

  async delete(key: string): Promise<void> {
    const p = this.pathOf(key);
    this.arriving.delete(key);
    await rm(p, { force: true });
    await rm(`${p}.parts.partial`, { force: true });
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
      if (seen > limit) done(tooLarge());
      else done(null, chunk);
    },
  });
}

export function registerBlobRoutes(app: FastifyInstance, ctx: AppContext): void {
  void app.register(async (scope) => {
    // Bodies stream straight to disk instead of being parsed.
    scope.removeAllContentTypeParsers();
    scope.addContentTypeParser("*", (_req, payload, done) => done(null, payload));

    type BlobRequest = {
      Params: { "*": string };
      Querystring: { exp?: string; sig?: string; part?: string; parts?: string };
    };

    /** Which part of how many a PUT carries: null for a whole blob, "bad" when it makes no sense. */
    const partOf = (q: BlobRequest["Querystring"]) => {
      if (q.part === undefined && q.parts === undefined) return null;
      if (!/^[0-9]{1,3}$/.test(q.part ?? "") || !/^[0-9]{1,3}$/.test(q.parts ?? "")) return "bad";
      const index = Number(q.part);
      const total = Number(q.parts);
      return total >= 1 && total <= MAX_PARTS && index < total ? { index, total } : "bad";
    };

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
      const part = partOf(req.query);
      if (part === "bad") return reply.code(400).send({ error: "Bad part." });
      try {
        // pipeline (unlike .pipe) passes a dropped connection on, so the store discards the file.
        const limited = sizeLimit(part ? MAX_PART_BYTES : MAX_BLOB_BYTES);
        pipeline(req.raw, limited).catch(() => {});
        if (part) {
          const result = await ctx.blobs.putPart(key, part.index, part.total, limited);
          if (result === "out-of-turn") return reply.code(409).send({ error: "Out of turn." });
          return reply.code(204).send();
        }
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
