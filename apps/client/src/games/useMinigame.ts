import type {
  Difficulty,
  GameId,
  MinigameJobResult,
  MinigameSettings,
  MinigameStep,
} from "@airace/protocol";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSocket } from "../store/session.ts";

export type Side = "a" | "b";

/** A run as the player watches it: each AI's steps and the words it is writing right now. */
export type LiveRun = {
  runId: string | null;
  status: "idle" | "starting" | "running" | "done" | "failed";
  steps: Record<Side, MinigameStep[]>;
  /** What each AI is typing, by `${side}:${attempt}`. */
  typing: Record<string, string>;
  result: MinigameJobResult | null;
  error: string | null;
};

const IDLE: LiveRun = {
  runId: null,
  status: "idle",
  steps: { a: [], b: [] },
  typing: {},
  result: null,
  error: null,
};

type StreamData =
  | { kind: "step"; side: Side; step: MinigameStep }
  | { kind: "text"; side: Side; attempt: number; text: string }
  | { kind: "done"; result: MinigameJobResult }
  | { kind: "failed" };

export type RunSummary = {
  id: string;
  game: GameId;
  difficulty: Difficulty;
  versionId: string;
  opponentName: string | null;
  showcase: boolean;
  status: string;
  score: number | null;
  maxScore: number | null;
  opponentScore: number | null;
  createdAt: number;
};

/** Plays and follows one minigame run in a world; also keeps the player's past runs and bests. */
export function useMinigame(worldId: string) {
  const socket = useSocket();
  const [run, setRun] = useState<LiveRun>(IDLE);
  const [history, setHistory] = useState<{ runs: RunSummary[]; bests: Record<string, number> }>({
    runs: [],
    bests: {},
  });
  const off = useRef<(() => void) | null>(null);

  const loadHistory = useCallback(() => {
    return socket
      .query<{ runs: RunSummary[]; bests: Record<string, number> }>({
        kind: "minigameRuns",
        worldId,
      })
      .then((h) => {
        setHistory(h);
        return h;
      })
      .catch(() => null);
  }, [socket, worldId]);
  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);
  useEffect(() => () => off.current?.(), []);

  /** Follows a run's live stream (a practice run, a duel or a showcase). */
  const follow = useCallback(
    (runId: string) => {
      off.current?.();
      setRun({ ...IDLE, runId, status: "running" });
      off.current = socket.subscribe(`minigame:${worldId}:${runId}`, (raw) => {
        const data = raw as StreamData;
        setRun((r) => {
          if (r.runId !== runId) return r;
          if (data.kind === "step") {
            const mine = r.steps[data.side].filter((s) => s.attempt !== data.step.attempt);
            return { ...r, steps: { ...r.steps, [data.side]: [...mine, data.step] } };
          }
          if (data.kind === "text") {
            const key = `${data.side}:${data.attempt}`;
            return { ...r, typing: { ...r.typing, [key]: (r.typing[key] ?? "") + data.text } };
          }
          if (data.kind === "done") {
            const b = data.result.sides.b?.steps ?? [];
            return {
              ...r,
              status: "done",
              result: data.result,
              steps: { a: data.result.sides.a.steps, b },
            };
          }
          return { ...r, status: "failed", error: "The game stopped. Try again in a moment." };
        });
        if (data.kind === "done" || data.kind === "failed") void loadHistory();
      });
    },
    [socket, worldId, loadHistory],
  );

  const start = useCallback(
    async (o: {
      game: GameId;
      difficulty: Difficulty;
      settings: MinigameSettings;
      versionId: string;
      opponent: string | null;
    }) => {
      setRun({ ...IDLE, status: "starting" });
      const r = await socket.minigame(worldId, o);
      if (!r.ok) {
        setRun({ ...IDLE, status: "failed", error: r.reason });
        return;
      }
      follow((r.data as { runId: string }).runId);
    },
    [socket, worldId, follow],
  );

  const reset = useCallback(() => {
    off.current?.();
    off.current = null;
    setRun(IDLE);
  }, []);

  return { run, start, follow, reset, history, loadHistory };
}
