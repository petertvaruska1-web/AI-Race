import { rmSync } from "node:fs";
import { EvalJobSpec } from "@airace/protocol";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it, vi } from "vitest";
import { schema } from "../src/db/index.ts";
import { eventually } from "./fakeWorker.ts";
import { closeSockets, setup, start, toDay, trainVersion, worker } from "./flows.ts";
import { api, createGuest, type TestServer } from "./harness.ts";
import { writeFixturePack } from "./rivalPack.ts";

let server: TestServer | null = null;
const temps: string[] = [];
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
  for (const t of temps.splice(0)) rmSync(t, { recursive: true, force: true });
});

function pack(identity?: string) {
  const p = writeFixturePack(identity);
  temps.push(p.dir);
  return p;
}

const rivalIds = (s: { companies: Record<string, { rival: unknown }> }) =>
  Object.entries(s.companies)
    .filter(([, c]) => c.rival !== null)
    .map(([id]) => id)
    .sort();

describe("the rival service", () => {
  it("seats four rivals in a running world once, also after a restart", async () => {
    const { dir } = pack();
    server = await start({ rivalsDir: dir });
    const o = await setup(server);
    await eventually(() => o.actor.state().rivals);
    expect(rivalIds(o.actor.state())).toEqual(["rival-1", "rival-2", "rival-3", "rival-4"]);
    await server.ctx.worlds.evict(o.worldId);
    const again = await server.ctx.worlds.get(o.worldId);
    await new Promise((r) => setTimeout(r, 50));
    expect(rivalIds(again.state())).toHaveLength(4);
  });

  it("seats rivals in an older world once the pack is built", async () => {
    server = await start();
    const o = await setup(server);
    await new Promise((r) => setTimeout(r, 50));
    expect(o.actor.state().rivals).toBeNull();
    const db = server.ctx.opts.dbDir;
    expect(db).toBeNull(); // in memory: a restart is a new server on the same database below
    const { dir } = pack();
    // Same process, new service: what a restart with the pack installed does.
    const { RivalService } = await import("../src/rivals/service.ts");
    server.ctx.opts.rivalsDir = dir;
    server.ctx.rivals = new RivalService(server.ctx);
    await server.ctx.worlds.evict(o.worldId);
    const again = await server.ctx.worlds.get(o.worldId);
    await eventually(() => again.state().rivals);
    expect(rivalIds(again.state())).toHaveLength(4);
  });

  it("every world draws from the pack of its rules; every rival AI is found in its own pack", async () => {
    const v1 = pack();
    const v2 = writeFixturePack("text-1-identity", "rivals-v2");
    const v3 = writeFixturePack(undefined, "rivals-v3");
    temps.push(v2.dir, v3.dir);
    server = await start({ rivalsDir: v3.dir, earlierRivalsDirs: [v2.dir, v1.dir] });
    const { rivals } = server.ctx;
    expect(rivals.packFor("balance-v2")?.dir).toBe(v1.dir);
    expect(rivals.packFor("balance-v3")?.dir).toBe(v2.dir);
    expect(rivals.packFor("balance-v4")?.dir).toBe(v3.dir);
    expect(rivals.packFor("balance-test")?.dir).toBe(v3.dir);
    // A rival AI's key says which pack it is in: its stored replies and its model come from there.
    for (const key of ["premium-a-v1", "premium-a-e2-v1", "premium-a-r3-v1"]) {
      expect(rivals.replies(key)[0]?.reply).toBe(`${key} answers students.`);
      const model = await fetch(await rivals.modelUrl(key));
      expect(await model.text()).toBe(`fake-rival-${key}`);
    }
    await expect(rivals.modelUrl("nobody-v9")).rejects.toThrow(/no rival pack holds/);
    // This server makes balance-test worlds: their rivals are the current pack's only.
    const o = await setup(server);
    await eventually(() => o.actor.state().rivals);
    const logged = JSON.stringify(await server.ctx.db.select().from(schema.worldInputs));
    expect(logged).toContain("premium-a-r3-v1");
    expect(logged).not.toContain('"premium-a-v1"');
    expect(logged).not.toContain("premium-a-e2-v1");
  });

  it("an earlier pack built on other data still serves the rivals seated from it", async () => {
    const v2 = writeFixturePack("text-1-identity", "rivals-v2");
    temps.push(v2.dir);
    server = await start({ rivalsDir: null, earlierRivalsDirs: [v2.dir] });
    expect(server.ctx.rivals.earlier.map((l) => l.dir)).toEqual([v2.dir]);
    expect(server.ctx.rivals.replies("premium-a-e2-v1")).not.toHaveLength(0);
  });

  it("a new world is never seated from an earlier pack", async () => {
    const v2 = writeFixturePack("text-1-identity", "rivals-v2");
    temps.push(v2.dir);
    server = await start({ rivalsDir: null, earlierRivalsDirs: [v2.dir] });
    expect(server.ctx.rivals.installed).toBe(false);
    expect(server.ctx.rivals.packFor("balance-test")).toBeNull();
    expect(server.ctx.rivals.packFor("balance-v4")).toBeNull();
    // A world of the rules the earlier pack was built for still draws from it.
    expect(server.ctx.rivals.packFor("balance-v3")?.dir).toBe(v2.dir);
    const o = await setup(server);
    await new Promise((r) => setTimeout(r, 50));
    expect(o.actor.state().rivals).toBeNull();
  });

  it("a world of older rules whose own pack is missing draws from the current pack", async () => {
    const v3 = writeFixturePack(undefined, "rivals-v3");
    temps.push(v3.dir);
    server = await start({ rivalsDir: v3.dir });
    for (const rules of ["balance-v2", "balance-v3", "balance-v4"] as const) {
      expect(server.ctx.rivals.packFor(rules)?.dir).toBe(v3.dir);
    }
  });

  it("refuses a pack whose AIs' keys another pack already uses", async () => {
    const first = pack("text-1-identity");
    const again = pack("text-1-identity");
    server = await start();
    const errors = vi.spyOn(server.ctx.log, "error");
    const { RivalService } = await import("../src/rivals/service.ts");
    server.ctx.opts.earlierRivalsDirs = [first.dir, again.dir];
    const rivals = new RivalService(server.ctx);
    expect(rivals.earlier.map((l) => l.dir)).toEqual([first.dir]);
    expect(errors).toHaveBeenCalled();
    expect(rivals.replies("premium-a-v1")[0]?.reply).toBe("premium-a-v1 answers students.");
  });

  it("runs without rivals when the pack is missing or was built from other data", async () => {
    const stale = pack("another-identity");
    server = await start({ rivalsDir: stale.dir });
    const o = await setup(server);
    await new Promise((r) => setTimeout(r, 50));
    expect(o.actor.state().rivals).toBeNull();
    const guest = await createGuest(server);
    const res = await api(server, "/api/setup", { cookie: guest.cookie });
    expect(await res.json()).toEqual({ trainingData: true, rivals: false });
  });

  it("answers a live rival AI's weekly feedback from its stored replies, with no job", async () => {
    const { dir } = pack();
    server = await start({ rivalsDir: dir });
    const o = await setup(server);
    await eventually(() => o.actor.state().rivals);
    const arch = () => o.actor.state().companies["rival-1"];
    const first = arch()?.rival?.upcoming[0];
    if (!first) throw new Error("nothing scheduled");
    await toDay(server, o.actor, first.releaseDay + 4);
    await eventually(() =>
      o.actor
        .state()
        .feed.some((f) => f.kind === "post" && f.companyId === "rival-1" && f.reply.length > 0),
    );
    const posts = o.actor
      .state()
      .feed.filter((f) => f.kind === "post" && f.companyId === "rival-1");
    for (const p of posts) {
      if (p.kind === "post")
        expect([`${first.key} answers ${p.author.group}.`, "<b>x</b> <|end|>"]).toContain(p.reply);
    }
    const jobs = await server.ctx.db
      .select()
      .from(schema.jobs)
      .where(eq(schema.jobs.type, "feedback"));
    expect(jobs).toHaveLength(0);
  });

  it("puts a rival AI's checkpoint in the blob store once and signs a URL for it", async () => {
    const { dir } = pack();
    server = await start({ rivalsDir: dir });
    const url = await server.ctx.rivals.modelUrl("premium-a-v1");
    expect(await (await fetch(url)).text()).toBe("fake-rival-premium-a-v1");
    const again = await server.ctx.rivals.modelUrl("premium-a-v1");
    expect(await (await fetch(again)).text()).toBe("fake-rival-premium-a-v1");
  });

  it("scores a player's AI against the rivals' AIs too", async () => {
    const { dir } = pack();
    server = await start({ rivalsDir: dir });
    const w = await worker(server, { jobTypes: ["train", "eval"] });
    const o = await setup(server);
    await eventually(() => o.actor.state().rivals);
    await trainVersion(server, w, o);
    const lease = await w.nextLease();
    expect(lease.jobType).toBe("eval");
    const spec = EvalJobSpec.parse(lease.spec);
    const rivalTraits = Object.values(o.actor.state().companies)
      .filter((c) => c.rival)
      .flatMap((c) => Object.values(c.versions).map((v) => v.traits));
    expect(rivalTraits.length).toBeGreaterThan(0);
    for (const t of rivalTraits) expect(spec.population).toContainEqual(t);
  });

  it("gives the same week's replies for the same batch", async () => {
    const { dir } = pack();
    server = await start({ rivalsDir: dir });
    const a = server.ctx.rivals.feedbackFor("p-1", "budget-a-v2", "w3", 1);
    const b = server.ctx.rivals.feedbackFor("p-1", "budget-a-v2", "w3", 1);
    expect(a).toEqual(b);
    expect(a).toHaveLength(6);
  });
});
