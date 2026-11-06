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
/** "Learn from customers" and "Train on your lessons": each adds this share of the run's text. */
export const COACHING_SHARE = 0.05;

export function toTrainRunConfig(
  plan: TrainingPlan,
  o: {
    runId: string;
    seed: number;
    balance: Balance;
    parent: { dir: string } | null;
    /** The conversations customers loved, for a plan that learns from them. */
    coaching?: [string, string][];
    /** The player's lessons, for a plan that trains on them (coaching). */
    lessons?: [string, string][];
    /** The notebook's paragraphs, for a plan with a notebook share. */
    notebook?: string[];
    /** The run's recorded `preview.tokens`: what the AI really reads. */
    tokens?: number;
  },
): Record<string, unknown> {
  const brain = o.balance.brainSizes.find((b) => b.id === plan.brainSize);
  const length = o.balance.lengths.find((l) => l.id === plan.length);
  if (!brain || !length) throw new Error(`plan names an unknown brain size or length`);
  let mixture: Record<string, number> = {};
  for (const id of DATASET_IDS) if (plan.mix[id] > 0) mixture[id] = plan.mix[id];
  // The notebook trains only with text in it (the world refuses a share without words).
  const notebook = plan.mix.notebook > 0 ? (o.notebook ?? []) : [];
  if (notebook.length > 0) mixture.notebook = plan.mix.notebook;
  // Tutoring packs: extra tokens on top of the length, all from one dataset (as `previewPlan`).
  const packs = plan.tutoring?.packs ?? 0;
  const budget = Math.round(length.tokens * (1 + o.balance.economy.tutoring.packShare * packs));
  const loved = plan.learnFromCustomers ? (o.coaching ?? []) : [];
  const lessons = plan.coaching ? (o.lessons ?? []) : [];
  const coaching = [...loved, ...lessons];
  // Each source adds its own share; together they are the trainer's one `coaching` dataset.
  const coachingShare = COACHING_SHARE * (Number(loved.length > 0) + Number(lessons.length > 0));
  if (packs > 0 || coaching.length > 0) {
    const total = Object.values(mixture).reduce((a, b) => a + b, 0);
    const own = 1 - coachingShare;
    const base = length.tokens / budget;
    mixture = Object.fromEntries(
      Object.entries(mixture).map(([k, w]) => [k, (own * base * w) / total]),
    );
    if (plan.tutoring && packs > 0) {
      const id = plan.tutoring.dataset;
      mixture[id] = (mixture[id] ?? 0) + own * (1 - base);
    }
    if (coaching.length > 0) mixture.coaching = coachingShare;
  }
  return {
    run_id: o.runId,
    seed: o.seed,
    shape: { n_layer: brain.nLayer, d_model: brain.dModel, ctx_len: plan.attentionSpan },
    // The run's recorded reading (Research makes it more than the paid length).
    token_budget: o.tokens ?? budget,
    mixture,
    ...(coaching.length > 0 ? { coaching } : {}),
    ...(notebook.length > 0 ? { notebook } : {}),
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
