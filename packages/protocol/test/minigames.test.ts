import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  MinigameSettings,
  QUIZ_TOPICS,
  STORY_HEROES,
  STORY_SITUATIONS,
  STORY_WORDS,
} from "../src/index.ts";

const shared = JSON.parse(
  readFileSync(new URL("../fixtures/minigames/faces.json", import.meta.url), "utf8"),
);

describe("minigames", () => {
  it("know the same Story Dice faces and quiz topics as the worker", () => {
    expect(STORY_HEROES).toEqual(shared.faces.hero);
    expect(STORY_SITUATIONS).toEqual(shared.faces.situation);
    expect(STORY_WORDS).toEqual(shared.faces.word);
    expect(QUIZ_TOPICS).toEqual(shared.quizTopics);
  });

  it("refuses dice faces that don't exist", () => {
    expect(MinigameSettings.safeParse({ rolls: [[0, 19, null]] }).success).toBe(true);
    expect(MinigameSettings.safeParse({ rolls: [[20, 0, 0]] }).success).toBe(false);
    expect(MinigameSettings.safeParse({ rolls: [[0, 0, 12]] }).success).toBe(false);
    expect(MinigameSettings.safeParse({ cheat: true }).success).toBe(false);
  });
});
