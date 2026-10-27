import { create } from "zustand";
import type { Me } from "../net/api.ts";
import type { GameSocket } from "../net/socket.ts";

type SessionStore = {
  me: Me | null;
  /** Set once the player is signed in; every screen talks through it. */
  socket: GameSocket | null;
  setMe(me: Me | null): void;
  setSocket(socket: GameSocket | null): void;
};

export const useSession = create<SessionStore>()((set) => ({
  me: null,
  socket: null,
  setMe: (me) => set({ me }),
  setSocket: (socket) => set({ socket }),
}));

/** The signed-in socket (screens render only after sign-in, so it is always there). */
export function useSocket(): GameSocket {
  const socket = useSession((s) => s.socket);
  if (!socket) throw new Error("not signed in");
  return socket;
}
