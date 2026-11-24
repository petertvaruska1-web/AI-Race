/**
 * The release's check of old worlds (plan C, Task 7 step 8): loads every world of a data root
 * through the real server code with the release's rival packs, and checks that every rival AI
 * already seated in a world still has its pack, its stored replies and its model.
 *
 * Run it on a COPY of the owner's worlds, never on `data/server` (loading a world writes to it):
 *
 *   npx tsx docs/handoff/release/check_worlds.ts <scratch data root>
 *
 * The scratch root holds `server/` (the copy) and links to the real `catalog`, `corpus`, `judge`,
 * `novelty`, `tokenizer` and `rivals` folders. Exit 0 when every world and every rival is fine.
 */
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { buildApp } from "../../../apps/server/src/app.ts";
import { schema } from "../../../apps/server/src/db/index.ts";
import { fakeScheduler } from "../../../apps/server/src/scheduler.ts";

const root = resolve(process.argv[2] ?? "");
if (!process.argv[2] || !existsSync(join(root, "server", "db"))) {
  console.error("usage: npx tsx docs/handoff/release/check_worlds.ts <scratch data root with server/db>");
  process.exit(2);
}
if (resolve(root) === resolve("data")) {
  console.error("refused: this check writes to the worlds it loads; run it on a copy");
  process.exit(2);
}

const { ctx, close } = await buildApp({
  dbDir: join(root, "server", "db"),
  blobDir: join(root, "server", "blobs"),
  catalogPath: join(root, "catalog", "v1", "catalog.json"),
  rivalsDir: join(root, "rivals", "v3"),
  earlierRivalsDirs: [join(root, "rivals", "v2"), join(root, "rivals", "v1")],
  secret: "release-check",
  workerToken: "release-check",
  balance: "balance-v4",
  // Nothing ticks: the check only loads and reads the worlds.
  scheduler: fakeScheduler(),
  publicUrl: "http://127.0.0.1:1",
  logLevel: "warn",
  dataRoot: root,
});

const { rivals } = ctx;
const packs = [rivals.current, ...rivals.earlier].filter((p) => p !== null);
console.log(`catalog data: ${ctx.catalog?.dataIdentity.slice(0, 12) ?? "none"}`);
for (const p of packs) {
  console.log(
    `pack ${p.pack.version} in ${p.dir}: ${p.keys.size} AIs, data ${p.pack.dataIdentity.slice(0, 12)}`,
  );
}
const newWorlds = rivals.packFor("balance-v4");
console.log(`new worlds draw from: ${newWorlds?.pack.version ?? "nothing"}`);

let problems = newWorlds?.pack.version === "rivals-v3" ? 0 : 1;
const worlds = await ctx.db.select().from(schema.worlds);
for (const w of worlds) {
  try {
    const actor = await ctx.worlds.get(w.id);
    const s = actor.state();
    const byPack = new Map<string, number>();
    const missing: string[] = [];
    for (const c of Object.values(s.companies)) {
      if (!c.rival) continue;
      for (const v of Object.values(c.versions)) {
        if (!v.packKey) continue;
        const from = packs.find((p) => p.keys.has(v.packKey as string));
        const ok =
          from !== undefined &&
          rivals.replies(v.packKey).length > 0 &&
          existsSync(join(from.dir, "models", `${v.packKey}.tar`));
        if (!ok) missing.push(v.packKey);
        else byPack.set(from.pack.version, (byPack.get(from.pack.version) ?? 0) + 1);
      }
    }
    const seated = [...byPack].map(([k, n]) => `${n} from ${k}`).join(", ") || "no rival AIs";
    const own = rivals.packFor(s.balance)?.pack.version ?? "none";
    console.log(
      `${missing.length ? "PROBLEM" : "ok     "} "${w.name}" (${s.balance}, ${s.phase}, day ${s.day}): ${seated}; its rules draw from ${own}${missing.length ? `; missing ${missing.join(", ")}` : ""}`,
    );
    problems += missing.length;
  } catch (e) {
    problems += 1;
    console.log(`PROBLEM "${w.name}": did not load (${(e as Error).message})`);
  }
}
await close();
console.log(problems === 0 ? `\nall ${worlds.length} worlds fine` : `\n${problems} problem(s)`);
process.exit(problems === 0 ? 0 : 1);
