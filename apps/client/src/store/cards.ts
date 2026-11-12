import { useEffect } from "react";
import { create } from "zustand";
import type { GameSocket } from "../net/socket.ts";
import { useSocket } from "./session.ts";

/** A graphics card as every player in the world sees it (never its key). */
export type CardView = {
  userId: string;
  playerName: string;
  gpuName: string | null;
  vramMb: number;
  /** × of the reference speed (an RTX 3050). */
  speed: number;
  online: boolean;
  /** Its helper is from an older version of the game: no work until it is set up again. */
  outdated?: boolean;
  /** The PC running the game; it counts as the world host's. */
  house: boolean;
};

/** The biggest brain the player's card (or Fair mode's shared cards) can train, and its speed. */
export type TrainingLimit = { maxBrain: string | null; tokensPerSecond: number };

type CardsStore = {
  worldId: string | null;
  /** Null until the server has answered. */
  cards: CardView[] | null;
  limit: TrainingLimit | null;
  /** Follows a world's cards (query, then the `cards:` stream); returns the stop function. */
  follow(socket: GameSocket, worldId: string): () => void;
};

export const useCards = create<CardsStore>()((set, get) => ({
  worldId: null,
  cards: null,
  limit: null,
  follow: (socket, worldId) => {
    if (get().worldId !== worldId) set({ worldId, cards: null, limit: null });
    const refresh = () => {
      socket
        .query<{ cards: CardView[] }>({ kind: "cards", worldId })
        .then((r) => get().worldId === worldId && set({ cards: r.cards }))
        .catch(() => {});
      socket
        .query<TrainingLimit>({ kind: "trainingLimit", worldId })
        .then((limit) => get().worldId === worldId && set({ limit }))
        .catch(() => {});
    };
    refresh();
    return socket.subscribe(`cards:${worldId}`, (data) => {
      const cards = (data as { cards?: CardView[] } | null)?.cards;
      if (cards && get().worldId === worldId) set({ cards });
      // The limit follows the cards that are online.
      socket
        .query<TrainingLimit>({ kind: "trainingLimit", worldId })
        .then((limit) => get().worldId === worldId && set({ limit }))
        .catch(() => {});
    });
  },
}));

/** Keeps the world's cards up to date while a world screen is open. */
export function useFollowCards(worldId: string): void {
  const socket = useSocket();
  const follow = useCards((s) => s.follow);
  useEffect(() => follow(socket, worldId), [socket, worldId, follow]);
}

/** "NVIDIA GeForce RTX 3050" → "RTX 3050"; the processor when there is no graphics card. */
export function shortCardName(gpuName: string | null): string {
  if (!gpuName || gpuName === "unknown") return "a computer";
  if (gpuName === "CPU") return "processor only";
  return gpuName.replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "");
}

/** The cards that train for `userId`: their own (Fair off), or everyone's (Fair on). */
export function usableCards(cards: CardView[], userId: string, fairMode: boolean): CardView[] {
  return cards.filter((c) => c.online && !c.outdated && (fairMode || c.userId === userId));
}
