import { DATASET_IDS, type TrainingPlan } from "@airace/protocol";
import type { Balance } from "@airace/sim";

/** Tokens per training step (ADR 0005: learns more per token than 16384 at early budgets). */
export const BATCH_TOKENS = 8192;

/**
 * The player's plan as the Python trainer's `TrainRunConfig` JSON (snake_case,
 * `ml/src/airace_ml/train/config.py`). Each plain-language control maps to one
 * real setting (R7). Fields Python defaults (purchases, notebook, coaching,
 * finishing mix) are left out until the game offers them (M3).
 */
export function toTrainRunConfig(
  plan: TrainingPlan,
  o: { runId: string; seed: number; balance: Balance; parent: { dir: string } | null },
): Record<string, unknown> {
  const brain = o.balance.brainSizes.find((b) => b.id === plan.brainSize);
  const length = o.balance.lengths.find((l) => l.id === plan.length);
  if (!brain || !length) throw new Error(`plan names an unknown brain size or length`);
  const mixture: Record<string, number> = {};
  for (const id of DATASET_IDS) if (plan.mix[id] > 0) mixture[id] = plan.mix[id];
  return {
    run_id: o.runId,
    seed: o.seed,
    shape: { n_layer: brain.nLayer, d_model: brain.dModel, ctx_len: plan.attentionSpan },
    token_budget: length.tokens,
    mixture,
    replay: plan.review,
    prep: {
      cleaning: plan.cleaning,
      dedup: plan.removeDuplicates,
      fact_check: plan.factCheck,
      variety: plan.variety,
    },
    boldness: plan.learningStyle,
    batch_tokens: BATCH_TOKENS,
    probe_prompts: plan.probePrompt ? [plan.probePrompt] : [],
    ...(o.parent ? { parent_dir: o.parent.dir } : {}),
  };
}

/** Training steps the trainer will run: `max(1, token_budget // batch_tokens)`, as in Python. */
export function totalSteps(tokens: number): number {
  return Math.max(1, Math.floor(tokens / BATCH_TOKENS));
}
