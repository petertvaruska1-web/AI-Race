import { describe, expect, it } from "vitest";
import {
  type CompanyState,
  dateOfDay,
  type NewsItem,
  type PostItem,
  projectView,
  step,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys, ticks, withRivals } from "./pack.ts";

/** c1 sells "Spark" (v1), whose measured personality has `words`. */
function launched(words: string[] = []): WorldState {
  const s = structuredClone(runningWorld());
  const c1 = s.companies.c1 as CompanyState;
  c1.versions.v1 = {
    id: "v1",
    label: "v1",
    runId: "r1",
    parentId: null,
    nLayer: 4,
    dModel: 128,
    ctxLen: 256,
    createdDay: s.day,
    trainedTokens: 1,
    scores: { language: 40 },
    overall: 40,
    personalityWords: words,
    traits: null,
    published: false,
    packKey: null,
  };
  const out = ok(
    step(
      s,
      cmd(
        {
          type: "launchProduct",
          versionId: "v1",
          name: "Spark",
          plans: ["free"],
          countdownDays: 3,
        },
        "p1",
      ),
    ),
  );
  return ticks(out, 3);
}

const answer = (score: number, reply = `reply ${score}`) => ({
  question: `Question ${score}?`,
  reply,
  score,
  checked: false,
});

const scored = (answers: unknown[], interviewId = "i1", productId = "p1") =>
  sys({ type: "interviewScored", interviewId, productId, answers });

const c1 = (s: WorldState) => s.companies.c1 as CompanyState;
const added = (before: WorldState, after: WorldState) => after.feed.slice(before.feed.length);

describe("interviews", () => {
  it("a strong one wins followers and a story, with a post quoting the best answer exactly", () => {
    const s0 = launched();
    c1(s0).fame.followers = 500;
    const hostile = "<script>x</script>  spaced\n<|end|>";
    const s = ok(step(s0, scored([answer(0.9, hostile), answer(0.8)])));
    const feed = added(s0, s);
    const news = feed.filter((f): f is NewsItem => f.kind === "news");
    const posts = feed.filter((f): f is PostItem => f.kind === "post");
    expect(news).toHaveLength(1);
    expect(news[0]?.topic).toBe("interview");
    expect(news[0]?.headline).toMatch(/^Spark wows in its interview with /);
    expect(posts).toHaveLength(1);
    expect(posts[0]?.reply).toBe(hostile);
    expect(posts[0]?.question).toBe("Question 0.9?");
    expect(posts[0]?.tone).toBe("praise");
    expect(posts[0]?.press).toBe(true);
    // r = 0.85: followers × (1 + 0.1 × 0.7)
    expect(c1(s).fame.followers).toBeCloseTo(500 * 1.07);
    const kept = c1(s).interviews[0];
    expect(kept).toMatchObject({ id: "i1", productId: "p1", productName: "Spark", day: s.day });
    expect(kept?.mean).toBeCloseTo(0.85);
    expect(kept?.answers[0]?.reply).toBe(hostile);
    expect(c1(s).lastInterviewDay).toBe(s.day);
  });

  it("a weak one with a big following goes viral with its worst answer", () => {
    const s0 = launched();
    c1(s0).fame.followers = 2_000;
    const s = ok(step(s0, scored([answer(0.5), answer(0.05, "gibberish")])));
    const feed = added(s0, s);
    const news = feed.filter((f): f is NewsItem => f.kind === "news");
    expect(news.map((n) => n.topic)).toEqual(["interview", "viral"]);
    expect(news[0]?.headline).toMatch(/^Spark stumbles through an interview with /);
    const posts = feed.filter((f): f is PostItem => f.kind === "post");
    expect(posts.map((p) => [p.tone, p.reply])).toEqual([
      ["praise", "reply 0.5"],
      ["roast", "gibberish"],
    ]);
  });

  it("a so-so one is told as such; the AI's measured personality moves the reaction", () => {
    const so = ok(step(launched(), scored([answer(0.5)])));
    const story = (s: WorldState) =>
      s.feed.filter((f): f is NewsItem => f.kind === "news" && f.topic === "interview")[0];
    expect(story(so)?.headline).toMatch(/gives a so-so interview to/);
    const warm = ok(step(launched(["warm", "bold", "inventive"]), scored([answer(0.6)])));
    expect(story(warm)?.headline).toMatch(/wows in its interview/); // 0.6 + 0.15
    const slips = ok(step(launched(["often slips", "timid"]), scored([answer(0.45)])));
    expect(story(slips)?.headline).toMatch(/stumbles through/); // 0.45 − 0.1
  });

  it("refuses what isn't a fair interview", () => {
    const s = launched();
    expect(rejected(step(s, scored([answer(1)], "i1", "p9")))).toBe("That AI isn't one of yours.");
    expect(rejected(step(s, scored([])))).toBe("That interview isn't valid.");
    expect(rejected(step(s, scored([1, 2, 3, 4, 5, 6].map(() => answer(1)))))).toBe(
      "That interview isn't valid.",
    );
    expect(rejected(step(s, scored([{ question: "q", reply: 3, score: 1, checked: true }])))).toBe(
      "That interview isn't valid.",
    );
    expect(rejected(step(s, scored([answer(2)])))).toBe("That interview isn't valid.");
    const retired = structuredClone(s);
    (c1(retired).products.p1 as { status: string }).status = "retired";
    expect(rejected(step(retired, scored([answer(1)])))).toBe("That AI isn't on sale.");
    const rivals = ticks(withRivals("balance-test"), 8);
    const rivalProduct = Object.values((rivals.companies["rival-1"] as CompanyState).products)[0];
    expect(rejected(step(rivals, scored([answer(1)], "i1", rivalProduct?.id ?? "x")))).toBe(
      "That AI isn't one of yours.",
    );
  });

  it("has a cooldown, and the same interview is published once", () => {
    let s = ok(step(launched(), scored([answer(0.7)])));
    expect(rejected(step(s, scored([answer(0.7)])))).toBe("Already published.");
    expect(rejected(step(s, scored([answer(0.7)], "i2")))).toBe(
      `You gave an interview recently. Next one on ${dateOfDay(s.day + 14)}.`,
    );
    expect(projectView(s, "u1").me.company?.nextInterviewDay).toBe(s.day + 14);
    s = ticks(s, 14);
    expect(step(s, scored([answer(0.7)], "i2")).ok).toBe(true);
  });

  it("keeps the last ten for their owner, answers verbatim", () => {
    let s = launched();
    for (let k = 0; k < 12; k++) {
      s = ok(step(s, scored([answer(0.5, `answer ${k}`)], `i${k}`)));
      s = ticks(s, 14);
    }
    const view = projectView(s, "u1").me.company;
    expect(view?.interviews).toHaveLength(10);
    expect(view?.interviews.at(-1)?.answers[0]?.reply).toBe("answer 11");
    expect(view?.interviews[0]?.outlet).toMatch(
      /Daily Byte|Circuit Weekly|TechTalk Radio|Model Times/,
    );
  });
});
