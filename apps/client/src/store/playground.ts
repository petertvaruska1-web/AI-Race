import { create } from "zustand";

type PlaygroundStore = {
  /** The version to talk to; null means the newest one. */
  versionId: string | null;
  setVersion(id: string | null): void;
};

export const usePlayground = create<PlaygroundStore>()((set) => ({
  versionId: null,
  setVersion: (versionId) => set({ versionId }),
}));
