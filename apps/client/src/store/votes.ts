import { useEffect } from "react";
import { create } from "zustand";
import type { GameSocket } from "../net/socket.ts";
import { useSocket } from "./session.ts";

/** A story round put to the vote: the prompt and two unlabelled stories. */
export type VoteRound = { round: number; prompt: string; left: string; right: string };
export type VoteTotals = { round: number; left: number; right: number }[];

/** A friends' story battle open to the other players' votes (Stage 2). */
export type Ballot = {
  battleId: string;
  rounds: VoteRound[];
  /** When voting closes, on this computer's clock. */
  closesAt: number;
  /** This player's picks so far, by round. */
  picks: Record<number, "left" | "right">;
  /** The totals once voting is over. */
  closed: VoteTotals | null;
};

type VotesStore = {
  worldId: string | null;
  ballots: Record<string, Ballot>;
  /** Follows `battlevote:<worldId>`; returns the stop function. */
  follow(socket: GameSocket, worldId: string): () => void;
  picked(battleId: string, round: number, pick: "left" | "right"): void;
};

type VoteMessage =
  | { kind: "open"; battleId: string; closesInMs: number; rounds: VoteRound[] }
  | { kind: "closed"; battleId: string; rounds: VoteTotals };

export const useVotes = create<VotesStore>()((set, get) => ({
  worldId: null,
  ballots: {},
  follow: (socket, worldId) => {
    if (get().worldId !== worldId) set({ worldId, ballots: {} });
    return socket.subscribe(`battlevote:${worldId}`, (data) => {
      const m = data as VoteMessage | null;
      if (!m || get().worldId !== worldId) return;
      if (m.kind === "open") {
        set((s) => ({
          ballots: {
            ...s.ballots,
            [m.battleId]: {
              battleId: m.battleId,
              rounds: m.rounds,
              closesAt: Date.now() + m.closesInMs,
              picks: {},
              closed: null,
            },
          },
        }));
      } else if (m.kind === "closed") {
        const b = get().ballots[m.battleId];
        if (b)
          set((s) => ({ ballots: { ...s.ballots, [m.battleId]: { ...b, closed: m.rounds } } }));
      }
    });
  },
  picked: (battleId, round, pick) =>
    set((s) => {
      const b = s.ballots[battleId];
      if (!b) return s;
      return {
        ballots: { ...s.ballots, [battleId]: { ...b, picks: { ...b.picks, [round]: pick } } },
      };
    }),
}));

/** Keeps the world's story votes up to date while a world screen is open. */
export function useFollowVotes(worldId: string): void {
  const socket = useSocket();
  const follow = useVotes((s) => s.follow);
  useEffect(() => follow(socket, worldId), [socket, worldId, follow]);
}
