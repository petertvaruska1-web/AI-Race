import { create } from "zustand";

const KEY = "airace.lighterGraphics";

function stored(): boolean {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

/**
 * "Lighter graphics" (design §11): a still glass and simpler 3D. The player switches it on the
 * top strip; the game switches it on by itself (`auto`) when frames stay slow. Kept per browser.
 */
export const useGraphics = create<{
  lighter: boolean;
  auto: boolean;
  setLighter(lighter: boolean, auto?: boolean): void;
}>((set) => ({
  lighter: stored(),
  auto: false,
  setLighter(lighter, auto = false) {
    // Only the player's own choice is kept: what the game switched on by itself lasts this visit.
    if (!auto) {
      try {
        localStorage.setItem(KEY, lighter ? "1" : "0");
      } catch {
        // A browser without storage still switches for this visit.
      }
    }
    set({ lighter, auto: lighter && auto });
  },
}));
