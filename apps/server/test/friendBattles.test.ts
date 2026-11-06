import { randomUUID } from "node:crypto";
import { BattleJobSpec } from "@airace/protocol";
import type { CompanyState } from "@airace/sim";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { BATTLE_OFFLINE, NOT_YOUR_VOTE, VOTE_MS } from "../src/battles/service.ts";
import { schema } from "../src/db/index.ts";
import { eventually, FakeWorker } from "./fakeWorker.ts";
import {
  closeSockets,
  openSockets,
  owner,
  send,
  setup,
  start,
  toDay,
  trainVersion,
  worker,
} from "./flows.ts";
import { api, createGuest, type TestServer } from "./harness.ts";
import { settle, type WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
});

const EVAL = {
  report: {
    suite: "bench-v1",
    scores: {
      knowledge: { score: 50, raw: 0.5, n: 40 },
      creativity: { score: 30, raw: 0.3, n: 40 },
    },
    overall: 40,
    missing: [],
    items: [],
    seconds: 1,
  },
  traits: { verbosity: 0.4 },
  personalityWords: [],
};
const JOBS = { jobTypes: ["train", "eval", "battle"] as ("train" | "eval" | "battle")[] };

type Player = {
  ws: WsClient;
  worldId: string;
  userId: string;
  cookie: string;
  actor: Awaited<ReturnType<typeof setup>>["actor"];
};

const companyOf = (p: Player) => {
  const s = p.actor.state();
  return s.companies[s.members[p.userId]?.companyId ?? ""] as CompanyState;
};

async function keyOf(s: TestServer, cookie: string): Promise<string> {
  const res = await api(s, "/api/helper/key", { cookie });
  return ((await res.json()) as { key: string }).key;
}

async function joinAs(s: TestServer, host: Player, name: string): Promise<Player> {
  const g = await createGuest(s, name);
  const res = await api(s, "/api/worlds/join", {
    body: { code: host.actor.state().inviteCode },
    cookie: g.cookie,
  });
  expect(res.status).toBe(200);
  const ws = await owner(s, g.cookie, host.worldId);
  return { ...g, ws, worldId: host.worldId, actor: host.actor };
}

/** Trains, scores and launches `p`'s AI on card `w`. */
async function onSale(s: TestServer, w: FakeWorker, p: Player): Promise<string> {
  const versionId = await trainVersion(s, w, p);
  const evalLease = await w.nextLease();
  expect(evalLease.jobType).toBe("eval");
  w.done(evalLease.jobId, EVAL);
  await eventually(() => companyOf(p).versions[versionId]?.scores);
  await send(p.ws, p.worldId, {
    type: "launchProduct",
    versionId,
    name: `${p.cookie.slice(0, 4)}AI`,
    plans: ["free"],
    countdownDays: 3,
  });
  const product = Object.values(companyOf(p).products)[0];
  if (!product) throw new Error("no product");
  return product.id;
}

/** Ana (host, house card) and Ben (his own card) each sell an AI. */
async function friends(s: TestServer) {
  const house = await worker(s, JOBS);
  const ana = await setup(s);
  await onSale(s, house, ana);
  const ben = await joinAs(s, ana, "Ben");
  await send(ben.ws, ben.worldId, {
    type: "createCompany",
    name: "Bolt",
    logo: { shape: "circle", symbol: "spark", initials: "BO" },
    palette: { primary: "#4f8cff", secondary: "#ffb020" },
  });
  const bensCard = await FakeWorker.connect(s, {
    ...JOBS,
    token: await keyOf(s, ben.cookie),
    workerId: "ben-pc",
  });
  openSockets.push(bensCard);
  const bensProduct = await onSale(s, bensCard, ben);
  await toDay(s, ana.actor, ana.actor.state().day + 4);
  return { house, ana, ben, bensCard, bensProduct };
}

async function challenge(p: Player, productId: string, skill: string): Promise<string | null> {
  const cid = randomUUID();
  p.ws.send({
    v: 1,
    type: "command",
    cid,
    worldId: p.worldId,
    cmd: { type: "startBattle", rivalProductId: productId, skill },
  });
  return Promise.race([
    p.ws.next("ack", (m) => m.cid === cid).then(() => null),
    p.ws.next("reject", (m) => m.cid === cid).then((m) => m.reason),
  ]);
}

async function vote(p: Player, battleId: string, round: number, pick: "left" | "right") {
  const cid = randomUUID();
  p.ws.send({ v: 1, type: "battleVote", cid, worldId: p.worldId, battleId, round, pick });
  return Promise.race([
    p.ws.next("ack", (m) => m.cid === cid).then(() => null),
    p.ws.next("reject", (m) => m.cid === cid).then((m) => m.reason),
  ]);
}

const side = (which: "a" | "b", k: number, score: number, story = false) => ({
  prompt: story ? "Write a story" : `Q${k}?`,
  expected: story ? null : `A${k}`,
  [which]: { reply: `${which === "a" ? "Ana's" : "Ben's"} answer ${k}`, score },
});

/** Each test trains, scores and launches two AIs first: slow under a full suite. */
const SLOW = 90_000;

describe("battles between friends' AIs", () => {
  it(
    "runs each side on its owner's card with only its own model, and both see one result",
    async () => {
      server = await start();
      const { house, ana, ben, bensCard, bensProduct } = await friends(server);
      // Each side's creativity knows only its own company's text.
      await server.ctx.db.insert(schema.lessons).values({
        worldId: ana.worldId,
        companyId: companyOf(ana).id,
        question: "Ana's question?",
        answer: "Ana's answer.",
        kind: "written",
        sourceThread: randomUUID(),
        sourceIdx: 1,
        at: 0,
      });
      expect(await challenge(ana, bensProduct, "knowledge")).toBeNull();
      const battle = await eventually(() =>
        companyOf(ana).battles.find((b) => b.status === "running"),
      );

      const a = await house.nextLease();
      const b = await bensCard.nextLease();
      const specA = BattleJobSpec.parse(a.spec);
      const specB = BattleJobSpec.parse(b.spec);
      expect([specA.sides, specB.sides]).toEqual([["a"], ["b"]]);
      expect(specA.b).toBeUndefined();
      expect(specB.a).toBeUndefined();
      expect(specA.seed).toBe(specB.seed);
      expect(specA.a?.ownText).toEqual(["Ana's question?", "Ana's answer."]);
      expect(specB.b?.ownText).toEqual([]);
      const db = server.ctx.db;
      const owners = await Promise.all(
        [a.jobId, b.jobId].map(async (id) => {
          const [row] = await db.select().from(schema.jobs).where(eq(schema.jobs.id, id));
          return row?.ownerUserId;
        }),
      );
      expect(owners).toEqual([ana.userId, ben.userId]);

      house.done(a.jobId, { rounds: [0, 1, 2].map((k) => side("a", k, 1)) });
      await settle(100);
      expect(companyOf(ana).battles.find((x) => x.id === battle.id)?.status).toBe("running");
      bensCard.done(b.jobId, { rounds: [0, 1, 2].map((k) => side("b", k, 0)) });
      const done = await eventually(() =>
        companyOf(ana).battles.find((x) => x.id === battle.id && x.status === "done"),
      );
      expect(done.outcome).toBe("win");
      expect(done.rounds?.[0]).toMatchObject({
        mine: { reply: "Ana's answer 0", score: 1 },
        theirs: { reply: "Ben's answer 0", score: 0 },
      });
      expect(companyOf(ben).battles.find((x) => x.id === battle.id)).toMatchObject({
        defending: true,
        outcome: "loss",
      });
    },
    SLOW,
  );

  it(
    "can't start while the defender's computer is off (Fair mode off)",
    async () => {
      server = await start();
      const { ana, bensCard, bensProduct } = await friends(server);
      bensCard.close();
      await eventually(() => server?.ctx.broker.connections().length === 1);
      expect(await challenge(ana, bensProduct, "knowledge")).toBe(BATTLE_OFFLINE);
    },
    SLOW,
  );

  it(
    "lets the other players vote on story rounds; the votes beat the judge",
    async () => {
      server = await start();
      const { house, ana, ben, bensCard, bensProduct } = await friends(server);
      const cy = await joinAs(server, ana, "Cy");
      const dee = await joinAs(server, ana, "Dee");
      const eli = await joinAs(server, ana, "Eli");
      const voters = [cy, dee, eli];
      for (const v of voters)
        v.ws.send({ v: 1, type: "subscribe", sub: `battlevote:${ana.worldId}` });
      expect(await challenge(ana, bensProduct, "creativity")).toBeNull();
      const battle = await eventually(() =>
        companyOf(ana).battles.find((b) => b.status === "running"),
      );
      const a = await house.nextLease();
      const b = await bensCard.nextLease();
      // The judge prefers Ben's stories every round.
      house.done(a.jobId, { rounds: [0, 1, 2].map((k) => side("a", k, 0.2, true)) });
      bensCard.done(b.jobId, { rounds: [0, 1, 2].map((k) => side("b", k, 0.9, true)) });

      const opened = (await cy.ws.next("stream", (m) => m.sub === `battlevote:${ana.worldId}`))
        .data as {
        kind: string;
        battleId: string;
        rounds: { round: number; prompt: string; left: string; right: string }[];
      };
      expect(opened.kind).toBe("open");
      expect(opened.rounds).toHaveLength(3);
      // Unlabelled: only the stories, never whose they are or how the judge scored them.
      expect(Object.keys(opened).sort()).toEqual(["battleId", "closesInMs", "kind", "rounds"]);
      for (const r of opened.rounds) {
        expect(Object.keys(r).sort()).toEqual(["left", "prompt", "right", "round"]);
      }
      const anasSide = (r: { left: string }) => (r.left.startsWith("Ana") ? "left" : "right");
      const other = (x: "left" | "right") => (x === "left" ? "right" : "left");

      expect(await vote(ben, battle.id, 0, "left")).toBe(NOT_YOUR_VOTE);
      expect(await vote(ana, battle.id, 0, "left")).toBe(NOT_YOUR_VOTE);
      for (const r of opened.rounds) {
        // Cy and Dee vote for Ana's story, Eli for Ben's: 2-1 every round.
        expect(await vote(cy, battle.id, r.round, anasSide(r))).toBeNull();
        expect(await vote(dee, battle.id, r.round, anasSide(r))).toBeNull();
        expect(await vote(eli, battle.id, r.round, other(anasSide(r)))).toBeNull();
      }
      const done = await eventually(() =>
        companyOf(ana).battles.find((x) => x.id === battle.id && x.status === "done"),
      );
      expect(done.outcome).toBe("win");
      expect(done.rounds?.map((r) => r.votes)).toEqual(
        [0, 1, 2].map(() => ({ mine: 2, theirs: 1 })),
      );
      const closed = (
        await cy.ws.next(
          "stream",
          (m) =>
            m.sub === `battlevote:${ana.worldId}` && (m.data as { kind: string }).kind === "closed",
        )
      ).data;
      expect(JSON.stringify(closed)).not.toMatch(/Cy|Dee|Eli|userId/);
    },
    SLOW,
  );

  it(
    "with no one else online, or no votes in time, the judge decides",
    async () => {
      server = await start();
      const { house, ana, bensCard, bensProduct } = await friends(server);
      expect(await challenge(ana, bensProduct, "creativity")).toBeNull();
      const first = await eventually(() =>
        companyOf(ana).battles.find((b) => b.status === "running"),
      );
      house.done((await house.nextLease()).jobId, {
        rounds: [0, 1, 2].map((k) => side("a", k, 0.2, true)),
      });
      bensCard.done((await bensCard.nextLease()).jobId, {
        rounds: [0, 1, 2].map((k) => side("b", k, 0.9, true)),
      });
      const judged = await eventually(() =>
        companyOf(ana).battles.find((x) => x.id === first.id && x.status === "done"),
      );
      expect(judged.outcome).toBe("loss");

      // A voter who never votes: the vote closes after five minutes and the judge decides.
      await joinAs(server, ana, "Cy");
      await toDay(server, ana.actor, ana.actor.state().day + 8); // past the battle cooldown
      expect(await challenge(ana, bensProduct, "creativity")).toBeNull();
      const second = await eventually(() =>
        companyOf(ana).battles.find((b) => b.status === "running"),
      );
      house.done((await house.nextLease()).jobId, {
        rounds: [0, 1, 2].map((k) => side("a", k, 0.2, true)),
      });
      bensCard.done((await bensCard.nextLease()).jobId, {
        rounds: [0, 1, 2].map((k) => side("b", k, 0.9, true)),
      });
      await settle(200);
      expect(companyOf(ana).battles.find((x) => x.id === second.id)?.status).toBe("running");
      await server.scheduler.advance(VOTE_MS);
      const late = await eventually(() =>
        companyOf(ana).battles.find((x) => x.id === second.id && x.status === "done"),
      );
      expect(late.outcome).toBe("loss");
      expect(late.rounds?.[0]?.votes).toBeUndefined();
    },
    SLOW,
  );
});
