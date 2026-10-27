import type { DatasetId, TrainingPlan } from "@airace/protocol";
import type { Balance } from "@airace/sim";
import { create } from "zustand";

type PlanStore = {
  worldId: string | null;
  plan: TrainingPlan | null;
  /** Starts a fresh plan from the world's defaults, growing from `parentVersionId`. */
  reset(worldId: string, balance: Balance, parentVersionId: string | null): void;
  update(change: Partial<TrainingPlan>): void;
  setWeight(id: DatasetId, weight: number): void;
};

/** The player's next training plan, kept while they move between the Lab's steps. */
export const usePlan = create<PlanStore>()((set, get) => ({
  worldId: null,
  plan: null,
  reset: (worldId, balance, parentVersionId) =>
    set({
      worldId,
      plan: { ...balance.defaultPlan, mix: { ...balance.defaultPlan.mix }, parentVersionId },
    }),
  update: (change) => {
    const plan = get().plan;
    if (plan) set({ plan: { ...plan, ...change } });
  },
  setWeight: (id, weight) => {
    const plan = get().plan;
    if (plan) set({ plan: { ...plan, mix: { ...plan.mix, [id]: weight } } });
  },
}));
