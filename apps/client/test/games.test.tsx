import type { MinigameStep } from "@airace/protocol";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import {
  BridgeStage,
  MazeStage,
  QuizStage,
  SimonStage,
  StoryDiceStage,
} from "../src/games/Stages.tsx";
import { Playground } from "../src/screens/Playground.tsx";
import { usePlayground } from "../src/store/playground.ts";
import { useSession } from "../src/store/session.ts";
import { liveGame } from "./mockSocket.ts";
import { SCORES, trainedWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;
beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  usePlayground.setState({ versionId: null });
});

const RUN = "22222222-2222-4222-8222-222222222222";

function step(over: Partial<MinigameStep> & { attempt: number }): MinigameStep {
  return {
    depth: over.attempt,
    retry: 0,
    itemId: `x-${over.attempt}`,
    band: 0,
    typed: false,
    prompt: "Question: Is Tom a fenk?\nAnswer:",
    display: { kind: "maze", question: "Is Tom a fenk?" },
    options: ["yes", "no"],
    answerIndex: 0,
    choice: 0,
    sure: [0.7, 0.3],
    reply: null,
    points: null,
    ruleMet: null,
    correct: true,
    lives: 3,
    ...over,
  };
}

/** Answers the latest minigameRuns query. */
async function answerRuns(bests: Record<string, number> = {}) {
  await waitFor(() =>
    expect(game.ws.sentOf("query").some((q) => q.q.kind === "minigameRuns")).toBe(true),
  );
  const q = game.ws
    .sentOf("query")
    .filter((m) => m.q.kind === "minigameRuns")
    .at(-1);
  game.ws.receive({ v: 1, type: "queryResult", id: q?.id, ok: true, data: { runs: [], bests } });
}

describe("Playground games", () => {
  it("offer five games, with personal bests", async () => {
    render(<Playground view={viewOf(trainedWorld(SCORES))} />);
    await userEvent.click(screen.getByRole("radio", { name: "Games" }));
    await answerRuns({ "bridge:easy": 7 });
    for (const name of ["Puzzle Maze", "Pattern Bridge", "Quiz Show", "Story Dice", "Simon Says"]) {
      expect(screen.getByRole("button", { name: `Play ${name}` })).toBeTruthy();
    }
    expect(await screen.findByTestId("best-bridge-easy")).toHaveProperty(
      "textContent",
      "Easy best: 7",
    );
  });

  it("start a practice run with the chosen settings and show it step by step", async () => {
    render(<Playground view={viewOf(trainedWorld(SCORES))} />);
    await userEvent.click(screen.getByRole("radio", { name: "Games" }));
    await answerRuns();
    await userEvent.click(screen.getByRole("button", { name: "Play Quiz Show" }));
    await userEvent.click(screen.getByRole("radio", { name: "Hard" }));
    fireEvent.change(screen.getByLabelText("Topic"), { target: { value: "science" } });
    await userEvent.click(screen.getByRole("button", { name: "Practice (free)" }));
    await waitFor(() => expect(game.ws.sentOf("minigame")).toHaveLength(1));
    const sent = game.ws.sentOf("minigame")[0];
    expect(sent).toMatchObject({
      game: "quiz",
      difficulty: "hard",
      settings: { topic: "science" },
      versionId: "v1",
      opponent: null,
    });
    game.ws.receive({ v: 1, type: "ack", cid: sent?.cid, duplicate: false, data: { runId: RUN } });
    await waitFor(() =>
      expect(game.ws.sentOf("subscribe").map((s) => s.sub)).toContain(`minigame:w1:${RUN}`),
    );
    const quiz = step({
      attempt: 0,
      display: {
        kind: "quiz",
        topic: "science",
        question: "What is H2O?",
        shown: ["water", "salt", "air", "gold"],
        struck: [],
      },
      options: ["water", "salt", "air", "gold"],
      sure: [0.6, 0.2, 0.1, 0.1],
    });
    game.ws.receive({
      v: 1,
      type: "stream",
      sub: `minigame:w1:${RUN}`,
      data: { kind: "step", side: "a", step: quiz },
    });
    expect(await screen.findByText("What is H2O?")).toBeTruthy();
    game.ws.receive({
      v: 1,
      type: "stream",
      sub: `minigame:w1:${RUN}`,
      data: { kind: "done", result: { sides: { a: { steps: [quiz], score: 1, maxScore: 12 } } } },
    });
    await answerRuns();
    expect(await screen.findByTestId("result")).toBeTruthy();
  });

  it("can't showcase an AI that isn't on sale, and says why", async () => {
    render(<Playground view={viewOf(trainedWorld(SCORES))} />);
    await userEvent.click(screen.getByRole("radio", { name: "Games" }));
    await answerRuns();
    await userEvent.click(screen.getByRole("button", { name: "Play Pattern Bridge" }));
    const show = screen.getByRole("button", { name: /Showcase live for the press \(\$20,000\)/ });
    expect(show).toHaveProperty("disabled", true);
    expect(screen.getByText("Put this AI on sale to show it off.")).toBeTruthy();
  });
});

describe("the stages", () => {
  const color = "#4f8cff";

  it("Puzzle Maze: the orb stands as deep as the right answers, misses marked", () => {
    const steps = [
      step({ attempt: 0 }),
      step({ attempt: 1, depth: 1, correct: false, choice: 1, lives: 2 }),
      step({ attempt: 2, depth: 1, retry: 1, lives: 2 }),
    ];
    render(<MazeStage steps={steps} typing={{}} color={color} />);
    expect(screen.getByRole("img", { name: "Puzzle maze, depth 2 of 10" })).toBeTruthy();
    expect(screen.getByText("🚪 yes")).toBeTruthy();
  });

  it("Pattern Bridge: a tile per right answer, a crumbling one for a miss", () => {
    const pic = { kind: "color", value: "red" };
    const bridge = (attempt: number, correct: boolean) =>
      step({
        attempt,
        correct,
        display: {
          kind: "bridge",
          terms: [pic, pic],
          options: [pic, { kind: "color", value: "blue" }],
        },
      });
    render(
      <BridgeStage
        steps={[bridge(0, true), bridge(1, true), bridge(2, false)]}
        typing={{}}
        color={color}
      />,
    );
    expect(screen.getAllByTestId("tile")).toHaveLength(2);
    expect(screen.getAllByTestId("crumbled")).toHaveLength(1);
    expect(screen.getAllByRole("img", { name: "red" }).length).toBeGreaterThan(0);
  });

  it("Quiz Show: struck answers, the ladder climbed", () => {
    const q = step({
      attempt: 0,
      display: {
        kind: "quiz",
        topic: "places",
        question: "Capital of Peru?",
        shown: ["Lima", "Quito", "Bogota", "Santiago"],
        struck: ["Bogota", "Santiago"],
      },
      options: ["Lima", "Quito"],
    });
    render(<QuizStage steps={[q]} typing={{}} color={color} />);
    expect(screen.getAllByTestId("struck")).toHaveLength(2);
    const ladder = screen.getByRole("list", { name: "Prize ladder" });
    expect(within(ladder).getAllByTestId("rung-won")).toHaveLength(1);
  });

  it("Story Dice: the dice, the AI's own story, the audience and the rule", () => {
    const s = step({
      attempt: 0,
      typed: true,
      options: null,
      choice: null,
      sure: null,
      display: { kind: "story", faces: [0, 1, 0], word: "key" },
      reply: "The fox found a key.",
      points: 8,
      audience: 0.5,
      ruleMet: true,
    });
    render(<StoryDiceStage steps={[s]} typing={{}} color={color} />);
    expect(screen.getByText("The fox found a key.")).toBeTruthy();
    expect(screen.getByTestId("rule").textContent).toBe('used "key" ✓');
    expect(screen.getByText("8 of 39 audience points")).toBeTruthy();
  });

  it("Simon Says: green for done right, red for not quite, the reply as written", () => {
    const simon = (attempt: number, correct: boolean, reply: string) =>
      step({
        attempt,
        correct,
        typed: true,
        options: null,
        choice: null,
        sure: null,
        reply,
        display: {
          kind: "simon",
          check: "one_word",
          instruction: "Answer in one word: hot or cold?",
        },
      });
    render(
      <SimonStage
        steps={[simon(0, true, "Hot."), simon(1, false, "<b>x</b>")]}
        typing={{}}
        color={color}
      />,
    );
    expect(screen.getAllByTestId("card-green")).toHaveLength(1);
    expect(screen.getAllByTestId("card-red")).toHaveLength(1);
    expect(screen.getByText("<b>x</b>")).toBeTruthy();
  });
});
