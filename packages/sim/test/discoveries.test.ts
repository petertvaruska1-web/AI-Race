import type { Discovery } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  type CompanyState,
  type NewsItem,
  type PostItem,
  projectView,
  step,
  type VersionState,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys } from "./pack.ts";

const HOSTILE = '<img src=x onerror="alert(1)"> spaced   out\n<|end|>';

const finding = (over: Partial<Discovery> = {}): Discovery => ({
  key: "knowledge/topic:animals",
  kind: "strength",
  about: "facts about animals",
  title: "Surprisingly good at facts about animals",
  category: "knowledge",
  tag: "topic:animals",
  value: 80,
  compared: 30,
  evidence: { prompt: "Which group does the newt belong to?", reply: HOSTILE },
  ...over,
});

function withVersion(s0: WorldState, id = "v1", company = "c1"): WorldState {
  const s = structuredClone(s0);
  (s.companies[company] as CompanyState).versions[id] = {
    id,
    label: "Spark 1",
    runId: `r-${id}`,
    parentId: null,
    nLayer: 2,
    dModel: 64,
    ctxLen: 64,
    createdDay: 0,
    trainedTokens: 1,
    scores: null,
    overall: null,
    personalityWords: [],
    traits: null,
    published: false,
    packKey: null,
  } as VersionState;
  return s;
}

const found = (versionId: string, discoveries: unknown[]) =>
  sys({ type: "discoveryFound", versionId, discoveries });

const c1 = (s: WorldState) => s.companies.c1 as CompanyState;

describe("discoveries", () => {
  it("are stored for their version, numbered, and found only once", () => {
    let s = withVersion(runningWorld());
    s = ok(step(s, found("v1", [finding(), finding({ kind: "weakness", key: "coding/fam:if" })])));
    expect(c1(s).discoveries.map((d) => [d.id, d.kind, d.published])).toEqual([
      ["v1-d0", "strength", false],
      ["v1-d1", "weakness", false],
    ]);
    expect(c1(s).discoveries[0]?.day).toBe(s.day);
    expect(rejected(step(s, found("v1", [finding()])))).toBe("Already found.");
  });

  it("refuse an unknown or rival version, too many findings and a bad shape", () => {
    const s = withVersion(runningWorld());
    expect(rejected(step(s, found("v9", [finding()])))).toBe("Unknown version.");
    expect(rejected(step(s, found("v1", [finding(), finding(), finding(), finding()])))).toMatch(
      /isn't valid/,
    );
    expect(rejected(step(s, found("v1", [{ ...finding(), kind: "amazing" }])))).toMatch(
      /isn't valid/,
    );
    expect(rejected(step(s, found("v1", [])))).toMatch(/isn't valid/);
  });

  it("keep only the last thirty", () => {
    let s = runningWorld();
    for (let k = 0; k < 11; k++) {
      s = withVersion(s, `v${k}`);
      s = ok(step(s, found(`v${k}`, [finding(), finding(), finding()])));
    }
    expect(c1(s).discoveries).toHaveLength(30);
    expect(c1(s).discoveries[0]?.id).toBe("v1-d0");
  });

  it("publish as one story and one post that quotes the AI byte for byte, once", () => {
    let s = withVersion(runningWorld());
    s = ok(step(s, found("v1", [finding()])));
    c1(s).fame.followers = 1_000;
    const before = s.feed.length;
    s = ok(step(s, cmd({ type: "publishDiscovery", discoveryId: "v1-d0" })));
    const added = s.feed.slice(before);
    const news = added.filter((f): f is NewsItem => f.kind === "news");
    const posts = added.filter((f): f is PostItem => f.kind === "post");
    expect(news.map((n) => [n.topic, n.headline, n.sentiment])).toEqual([
      ["discovery", "Acme's Spark 1 turns out to be surprisingly good at facts about animals", 0.4],
    ]);
    expect(posts).toHaveLength(1);
    expect(posts[0]?.question).toBe("Which group does the newt belong to?");
    expect(posts[0]?.reply).toBe(HOSTILE);
    expect(posts[0]?.author.group).toBe("researchers");
    expect(posts[0]?.tone).toBe("praise");
    expect(c1(s).fame.followers).toBeCloseTo(1_030);
    expect(c1(s).discoveries[0]?.published).toBe(true);
    expect(rejected(step(s, cmd({ type: "publishDiscovery", discoveryId: "v1-d0" })))).toBe(
      "You've already published that.",
    );
  });

  it("tell each kind in its own words; a finding without evidence makes no post", () => {
    const cases: [Discovery["kind"], string, number][] = [
      ["weakness", "Acme admits its Spark 1 struggles with facts about animals", 0.1],
      ["better", "Acme's Spark 1 got much better at facts about animals", 0.4],
      ["worse", "Acme's Spark 1 got worse at facts about animals", 0.1],
      ["changed", "Acme's Spark 1 has become more facts about animals", 0.2],
    ];
    for (const [kind, headline, sentiment] of cases) {
      let s = withVersion(runningWorld());
      s = ok(step(s, found("v1", [finding({ kind, evidence: null })])));
      const before = s.feed.length;
      s = ok(step(s, cmd({ type: "publishDiscovery", discoveryId: "v1-d0" })));
      const added = s.feed.slice(before);
      expect(added).toHaveLength(1);
      expect(added[0]).toMatchObject({ kind: "news", headline, sentiment });
    }
  });

  it("can't be published by anyone else, and are seen only by their owner", () => {
    let s = withVersion(runningWorld());
    s = ok(step(s, sys({ type: "memberJoined", userId: "u2", displayName: "Ben" })));
    s = ok(
      step(
        s,
        cmd(
          {
            type: "createCompany",
            name: "Beta",
            logo: { shape: "circle", symbol: "spark", initials: "BE" },
            palette: { primary: "#3a86ff", secondary: "#ffbe0b" },
          },
          "c2",
          "u2",
        ),
      ),
    );
    s = ok(step(s, found("v1", [finding()])));
    expect(
      rejected(step(s, cmd({ type: "publishDiscovery", discoveryId: "v1-d0" }, undefined, "u2"))),
    ).toBe("That discovery isn't yours.");
    const mine = JSON.stringify(projectView(s, "u1"));
    expect(mine.split("Surprisingly good at facts about animals")).toHaveLength(2);
    expect(projectView(s, "u1").me.company?.discoveries[0]?.evidence?.reply).toBe(HOSTILE);
    expect(JSON.stringify(projectView(s, "u2"))).not.toContain("Surprisingly good");
  });
});
