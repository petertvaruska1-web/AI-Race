import { applyPatch, type ServerMessage, type WorldView } from "@airace/protocol";
import { create } from "zustand";
import type { GameSocket, SocketStatus } from "../net/socket.ts";

type Snapshot = Extract<ServerMessage, { type: "snapshot" }>;
type Patch = Extract<ServerMessage, { type: "patch" }>;

export type WorldStore = {
  worldId: string | null;
  version: number;
  view: WorldView | null;
  status: SocketStatus;
  onSnapshot(m: Snapshot): void;
  /** Applies a patch that follows the current version; false (nothing applied) when one was missed. */
  onPatch(m: Patch): boolean;
  setStatus(s: SocketStatus): void;
  open(worldId: string | null): void;
};

/** The watched world as this player sees it: the server's view, kept current by patches. */
export const useWorld = create<WorldStore>()((set, get) => ({
  worldId: null,
  version: -1,
  view: null,
  status: "offline",
  onSnapshot: (m) => {
    if (m.worldId !== get().worldId) return;
    set({ version: m.version, view: m.view });
  },
  onPatch: (m) => {
    const { worldId, version, view } = get();
    if (m.worldId !== worldId) return true;
    if (!view || m.from !== version) return false;
    set({ version: m.to, view: applyPatch(view, m.ops) });
    return true;
  },
  setStatus: (status) => set({ status }),
  open: (worldId) => set({ worldId, version: -1, view: null }),
}));

/** Feeds the socket's world messages into the store; a missed patch asks for a fresh snapshot. */
export function bindWorldStore(socket: GameSocket, store = useWorld): () => void {
  const offWorld = socket.onWorld((m) => {
    const s = store.getState();
    if (m.type === "snapshot") s.onSnapshot(m);
    else if (m.type === "patch" && !s.onPatch(m)) socket.resync(m.worldId);
  });
  const offStatus = socket.onStatus((status) => store.getState().setStatus(status));
  return () => {
    offWorld();
    offStatus();
  };
}
