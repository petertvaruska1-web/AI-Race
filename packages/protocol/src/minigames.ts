import { z } from "zod";

// Playground minigames (playtest pack §4). The ML worker plays them with the real AIs
// (ml/src/airace_ml/minigames/); these are the shapes the server and client share.

export const GAMES = ["maze", "bridge", "quiz", "story", "simon"] as const;
export const GameId = z.enum(GAMES);
export type GameId = z.infer<typeof GameId>;

export const GAME_NAMES: Record<GameId, string> = {
  maze: "Puzzle Maze",
  bridge: "Pattern Bridge",
  quiz: "Quiz Show",
  story: "Story Dice",
  simon: "Simon Says",
};

/** Games a duel can be: every one but Story Dice (a story has no right answer to race to). */
export const DUEL_GAMES = ["maze", "bridge", "quiz", "simon"] as const satisfies readonly GameId[];

export const DIFFICULTIES = ["easy", "medium", "hard"] as const;
export const Difficulty = z.enum(DIFFICULTIES);
export type Difficulty = z.infer<typeof Difficulty>;

/** Steps in a full run (a story run is 3 stories). */
export const GAME_STEPS: Record<GameId, number> = {
  maze: 10,
  bridge: 12,
  quiz: 12,
  story: 3,
  simon: 10,
};
export const GAME_LIVES = 3;
/** Story Dice: 10 points for the audience's applause and 3 for a rule met, per story. */
export const STORY_MAX_POINTS = 39;

export const QUIZ_TOPICS = [
  "mixed",
  "animals",
  "places",
  "school",
  "science",
  "technology",
] as const;
export type QuizTopic = (typeof QUIZ_TOPICS)[number];

/** The Story Dice faces, in the order the worker numbers them (fixtures/minigames/faces.json). */
export const STORY_HEROES = [
  "a fox",
  "a shy robot",
  "an old lighthouse keeper",
  "a small dragon",
  "a lost puppy",
  "a brave mouse",
  "a sleepy bear",
  "a young painter",
  "a talking kettle",
  "a curious girl",
  "a grumpy goat",
  "a paper boat",
  "a tiny snail",
  "a busy bee",
  "a friendly ghost",
  "a clever crow",
  "a quiet boy",
  "a lonely moon",
  "a happy whale",
  "a forgetful wizard",
] as const;
export const STORY_SITUATIONS = [
  "who loses the way home",
  "who finds a door in a tree",
  "who wants to see the ocean",
  "who is afraid of the rain",
  "who makes a new friend",
  "who learns to whistle",
  "who wins a race by accident",
  "who finds a map in a bottle",
  "who misses the last train",
  "who builds a tiny house",
  "who hears a strange song",
  "who wakes up in the snow",
  "who tries to catch a star",
  "who plants a magic bean",
  "who loses a favorite hat",
  "who travels to the city",
  "who keeps a big secret",
  "who bakes a cake for the king",
  "who follows a rainbow",
  "who cannot fall asleep",
] as const;
export const STORY_WORDS = [
  "key",
  "map",
  "hat",
  "cake",
  "star",
  "boat",
  "song",
  "door",
  "tree",
  "moon",
  "bell",
  "ring",
] as const;

const Face = (n: number) =>
  z
    .number()
    .int()
    .min(0)
    .max(n - 1);

/** What the player sets before a run. */
export const MinigameSettings = z.strictObject({
  topic: z.enum(QUIZ_TOPICS).optional(),
  /** Story Dice: up to three rolls (hero, situation, word or none); missing ones are rolled. */
  rolls: z
    .array(
      z.tuple([
        Face(STORY_HEROES.length),
        Face(STORY_SITUATIONS.length),
        Face(STORY_WORDS.length).nullable(),
      ]),
    )
    .max(3)
    .optional(),
  /** Story Dice: whether rolled stories must use a word. */
  rule: z.boolean().optional(),
});
export type MinigameSettings = z.infer<typeof MinigameSettings>;

/** One attempt of one AI: what it was asked, what it did (its own choice or words, R1), and how it went. */
export const MinigameStep = z.object({
  attempt: z.number().int().min(0),
  depth: z.number().int().min(0),
  retry: z.number().int().min(0),
  itemId: z.string(),
  band: z.number().int().min(0),
  typed: z.boolean(),
  prompt: z.string(),
  /** What the client draws (the puzzle's words, the pattern's pictures, the dice...). */
  display: z.record(z.string(), z.unknown()),
  options: z.array(z.string()).nullable(),
  answerIndex: z.number().int().min(0).nullable(),
  choice: z.number().int().min(0).nullable(),
  /** How sure the AI was of each option (sums to 1). */
  sure: z.array(z.number().min(0).max(1)).nullable(),
  /** The AI's own words, exactly as written (R1). */
  reply: z.string().nullable(),
  /** Story Dice: the story's points, the judge's applause (0-1) and whether the rule was met. */
  points: z.number().nullable(),
  audience: z.number().min(0).max(1).optional(),
  ruleMet: z.boolean().nullable(),
  correct: z.boolean(),
  lives: z.number().int().min(0),
});
export type MinigameStep = z.infer<typeof MinigameStep>;

export const MinigameSideResult = z.object({
  steps: z.array(MinigameStep),
  score: z.number().min(0),
  maxScore: z.number().positive(),
});
export type MinigameSideResult = z.infer<typeof MinigameSideResult>;
