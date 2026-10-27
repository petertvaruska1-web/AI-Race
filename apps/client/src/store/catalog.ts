import type { PublicCatalog } from "@airace/protocol";
import { create } from "zustand";
import type { GameSocket } from "../net/socket.ts";

type CatalogStore = {
  catalog: PublicCatalog | null;
  error: string | null;
  loading: boolean;
  /** Asks the server for the dataset catalog once. */
  load(socket: GameSocket): void;
};

export const useCatalog = create<CatalogStore>()((set, get) => ({
  catalog: null,
  error: null,
  loading: false,
  load: (socket) => {
    if (get().catalog || get().loading) return;
    set({ loading: true, error: null });
    socket
      .query<PublicCatalog>({ kind: "catalog" })
      .then((catalog) => set({ catalog, loading: false }))
      .catch((err: Error) => set({ error: err.message, loading: false }));
  },
}));
