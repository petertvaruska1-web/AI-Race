import type { CostCategory, GroupId, IncomeCategory, RegionId } from "@airace/protocol";
import { create } from "zustand";

/** What a window explains (immersion design: every number has a story). */
export type WindowSpec =
  | { kind: "customers" }
  | { kind: "money" }
  | { kind: "group"; group: GroupId }
  | { kind: "product"; productId: string }
  | { kind: "ai"; versionId: string }
  | { kind: "month"; month: string }
  | { kind: "income"; category: IncomeCategory }
  | { kind: "cost"; category: CostCategory }
  | { kind: "post"; postId: string }
  | { kind: "region"; region: RegionId };

export type OpenWindow = {
  id: number;
  spec: WindowSpec;
  /** The middle of what was clicked, so the window grows out of it. */
  from: { x: number; y: number } | null;
  /** What opened it; it gets the focus back when the windows close. */
  opener: HTMLElement | null;
};

/** Windows opened from windows stack up; the oldest go beyond this many. */
const MAX_STACK = 8;

type Windows = {
  stack: OpenWindow[];
  seq: number;
  open: (spec: WindowSpec, opener?: Element | null) => void;
  /** Closes the top window, showing the one below. */
  back: () => void;
  closeAll: () => void;
};

export const useWindows = create<Windows>((set) => ({
  stack: [],
  seq: 0,
  open: (spec, opener = null) =>
    set((s) => {
      const r = opener?.getBoundingClientRect();
      const from =
        r && (r.width > 0 || r.height > 0)
          ? { x: r.left + r.width / 2, y: r.top + r.height / 2 }
          : null;
      const entry: OpenWindow = {
        id: s.seq + 1,
        spec,
        from,
        opener: opener instanceof HTMLElement ? opener : null,
      };
      return { seq: entry.id, stack: [...s.stack, entry].slice(-MAX_STACK) };
    }),
  back: () => set((s) => ({ stack: s.stack.slice(0, -1) })),
  closeAll: () => set({ stack: [] }),
}));
