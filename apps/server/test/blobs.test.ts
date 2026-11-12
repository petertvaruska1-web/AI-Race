import { randomBytes } from "node:crypto";
import { readdirSync } from "node:fs";
import { request } from "node:http";
import { join } from "node:path";
import { Readable } from "node:stream";
import { afterEach, describe, expect, it } from "vitest";
import { assertBlobKey, signBlobUrl } from "../src/blobs.ts";
import { startTestServer, type TestServer } from "./harness.ts";
import { settle } from "./wsClient.ts";

let server: TestServer | null = null;
afterEach(async () => {
  await server?.close();
  server = null;
});

describe("blob store", () => {
  it("round-trips 3 MB through signed PUT and GET URLs", async () => {
    server = await startTestServer();
    const data = randomBytes(3 * 1024 * 1024);
    const put = await fetch(signBlobUrl(server.ctx, "checkpoints/v1/checkpoint.tar", "PUT"), {
      method: "PUT",
      headers: { "content-type": "application/octet-stream" },
      body: data,
    });
    expect(put.status).toBe(204);
    const get = await fetch(signBlobUrl(server.ctx, "checkpoints/v1/checkpoint.tar", "GET"));
    expect(get.status).toBe(200);
    expect(Buffer.from(await get.arrayBuffer()).equals(data)).toBe(true);
    expect(await server.ctx.blobs.exists("checkpoints/v1/checkpoint.tar")).toBe(true);
  });

  it("takes a big file in parts, and the key is absent until the last part is in", async () => {
    server = await startTestServer();
    const key = "checkpoints/big/checkpoint.tar";
    const pieces = [randomBytes(300_000), randomBytes(300_000), randomBytes(1234)];
    const url = signBlobUrl(server.ctx, key, "PUT");
    const send = (i: number) =>
      fetch(`${url}&part=${i}&parts=${pieces.length}`, {
        method: "PUT",
        body: pieces[i] as Buffer,
      });
    expect((await send(0)).status).toBe(204);
    expect((await send(1)).status).toBe(204);
    expect(await server.ctx.blobs.exists(key)).toBe(false);
    expect((await send(2)).status).toBe(204);
    const get = await fetch(signBlobUrl(server.ctx, key, "GET"));
    expect(Buffer.from(await get.arrayBuffer()).equals(Buffer.concat(pieces))).toBe(true);
    // Nothing but the blob is left in its folder.
    expect(readdirSync(join(server.ctx.opts.blobDir, "checkpoints/big"))).toEqual([
      "checkpoint.tar",
    ]);
  });

  it("a part sent twice is kept once, a part out of turn is refused, and part 0 starts over", async () => {
    server = await startTestServer();
    const key = "resume/run.tar";
    const pieces = [randomBytes(50_000), randomBytes(50_000), randomBytes(50_000)];
    const url = signBlobUrl(server.ctx, key, "PUT");
    const send = (i: number, parts = pieces.length) =>
      fetch(`${url}&part=${i}&parts=${parts}`, { method: "PUT", body: pieces[i] as Buffer });
    expect((await send(1)).status).toBe(409); // nothing was started
    expect((await send(0)).status).toBe(204);
    expect((await send(2)).status).toBe(409); // part 1 is next
    expect((await send(1)).status).toBe(204);
    expect((await send(1)).status).toBe(204); // its answer was lost and it came again
    expect((await send(1, 5)).status).toBe(409); // another upload's part
    // The sender starts over (a 409 tells it to): part 0 always may.
    for (const i of [0, 1, 2]) expect((await send(i)).status).toBe(204);
    const get = await fetch(signBlobUrl(server.ctx, key, "GET"));
    expect(Buffer.from(await get.arrayBuffer()).equals(Buffer.concat(pieces))).toBe(true);
    for (const bad of ["part=3&parts=3", "part=0&parts=0", "part=x&parts=3", "part=0"]) {
      expect((await fetch(`${url}&${bad}`, { method: "PUT", body: "x" })).status, bad).toBe(400);
    }
  });

  it("refuses a URL signed for another method, an expired one and an unsigned one", async () => {
    server = await startTestServer();
    await server.ctx.blobs.put("a/b", Readable.from([Buffer.from("hi")]));
    const getUrl = signBlobUrl(server.ctx, "a/b", "GET", 60_000);
    expect((await fetch(getUrl, { method: "PUT", body: "x" })).status).toBe(403);
    expect((await fetch(`${server.url}/blobs/a/b`)).status).toBe(403);
    expect((await fetch(getUrl.replace("/a/b?", "/a/c?"))).status).toBe(403);
    expect((await fetch(getUrl)).status).toBe(200);
    await server.scheduler.advance(61_000);
    expect((await fetch(getUrl)).status).toBe(403);
    expect((await fetch(signBlobUrl(server.ctx, "a/missing", "GET"))).status).toBe(404);
  });

  it("rejects keys that could leave the store", async () => {
    server = await startTestServer();
    for (const key of ["../x", "a//..", "a/../b", "/abs", "A/upper", "a/./b", ""]) {
      expect(() => assertBlobKey(key), key).toThrow();
      expect(() => signBlobUrl(server?.ctx as never, key, "GET"), key).toThrow();
    }
    await expect(server.ctx.blobs.put("../x", Readable.from([Buffer.from("x")]))).rejects.toThrow();
    // URL parsing folds an encoded ".." away before the request leaves; it never reaches a blob.
    const url = signBlobUrl(server.ctx, "ok", "GET").replace("/blobs/ok?", "/blobs/%2e%2e/ok?");
    expect([403, 404]).toContain((await fetch(url)).status);
  });

  it("leaves nothing at the key when an upload stops halfway", async () => {
    server = await startTestServer();
    const url = new URL(signBlobUrl(server.ctx, "half/checkpoint.tar", "PUT"));
    await new Promise<void>((resolve) => {
      const req = request(url, {
        method: "PUT",
        headers: { "content-type": "application/octet-stream", "content-length": String(4 << 20) },
      });
      req.on("error", () => resolve());
      req.write(randomBytes(1 << 20), () => {
        setTimeout(() => {
          req.destroy();
          resolve();
        }, 50);
      });
    });
    await settle(200);
    expect(await server.ctx.blobs.exists("half/checkpoint.tar")).toBe(false);
    // The upload did reach the server (its folder exists), and its partial file goes soon after;
    // under a busy test run the cleanup can take longer than the 200 ms above.
    const folder = join(server.ctx.opts.blobDir, "half");
    for (let i = 0; i < 40 && readdirSync(folder).length > 0; i++) await settle(100);
    expect(readdirSync(folder)).toEqual([]);
  });
});
