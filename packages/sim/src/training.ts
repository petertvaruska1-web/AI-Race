import type { TrainingPlan, TrainingPreview } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { paramCount, trainingFlops } from "./shape.ts";
import type { CompanyState, RunState, TrainingResult, WorldState } from "./state.ts";

export type PreviewResult = { ok: true; preview: TrainingPreview } | { ok: false; reason: string };

/** What a plan costs in money and game days, or why it can't run. */
export function previewTraining(
  state: WorldState,
  companyId: string,
  plan: TrainingPlan,
): PreviewResult {
  const b = getBalance(state.balance);
  const company = state.companies[companyId];
  if (!company) return { ok: false, reason: "Unknown company." };
  if (Object.values(plan.mix).every((w) => w === 0)) {
    return { ok: false, reason: "Pick at least one dataset to train on." };
  }
  const brain = b.brainSizes.find((s) => s.id === plan.brainSize);
  if (!brain) return { ok: false, reason: "Unknown brain size." };
  const length = b.lengths.find((l) => l.id === plan.length);
  if (!length) return { ok: false, reason: "Unknown training length." };
  if (!b.attentionSpans.includes(plan.attentionSpan))
    return { ok: false, reason: "Unknown attention span." };
  if (plan.parentVersionId !== null) {
    const parent = company.versions[plan.parentVersionId];
    if (!parent) return { ok: false, reason: "That AI isn't one of yours." };
    // Growth keeps what the parent learned, so a brain can only grow (R6).
    if (brain.nLayer < parent.nLayer || brain.dModel < parent.dModel) {
      return {
        ok: false,
        reason: "A new version can't have a smaller brain than the AI it grows from.",
      };
    }
  }
  const params = paramCount(brain.nLayer, brain.dModel);
  const flops = trainingFlops(params, length.tokens);
  const pflop = flops / 1e15;
  return {
    ok: true,
    preview: {
      params,
      tokens: length.tokens,
      flops,
      cost: Math.round(pflop * b.cloudPricePerPflop) + (plan.factCheck ? b.factCheckCost : 0),
      durationDays: Math.max(1, Math.ceil(pflop / b.cloudPflopsPerDay)),
    },
  };
}

/** The company's training or finalizing run, if any. */
export function activeRun(company: CompanyState): RunState | null {
  for (const id of Object.keys(company.runs).sort()) {
    const run = company.runs[id];
    if (run && (run.status === "training" || run.status === "finalizing")) return run;
  }
  return null;
}

/** How far through its game time a run is, 0 to 1. */
export function runProgress(run: RunState, day: number): number {
  if (run.endDay <= run.startDay) return 1;
  return Math.min(1, Math.max(0, (day - run.startDay) / (run.endDay - run.startDay)));
}

export function findRun(
  state: WorldState,
  runId: string,
): { company: CompanyState; run: RunState } | null {
  for (const id of Object.keys(state.companies).sort()) {
    const company = state.companies[id];
    const run = company?.runs[runId];
    if (company && run) return { company, run };
  }
  return null;
}

/** Turns a finished run into the company's next version. Mutates `state`. */
export function applyCompletion(
  state: WorldState,
  company: CompanyState,
  run: RunState,
  versionId: string,
  _result: TrainingResult,
): void {
  const brain = getBalance(state.balance).brainSizes.find((s) => s.id === run.plan.brainSize);
  if (!brain) throw new Error(`run ${run.id} names an unknown brain size`);
  const parent =
    run.plan.parentVersionId === null ? null : company.versions[run.plan.parentVersionId];
  company.versionSeq += 1;
  company.versions[versionId] = {
    id: versionId,
    label: `v${company.versionSeq}`,
    runId: run.id,
    parentId: parent?.id ?? null,
    nLayer: brain.nLayer,
    dModel: brain.dModel,
    ctxLen: run.plan.attentionSpan,
    createdDay: state.day,
    trainedTokens: run.preview.tokens + (parent?.trainedTokens ?? 0),
    scores: null,
    overall: null,
    personalityWords: [],
    published: false,
  };
  run.status = "completed";
  run.versionId = versionId;
  run.held = null;
}

/** Applies held results whose game time has come; others start finalizing. Mutates `state`. */
export function settleRuns(state: WorldState): void {
  for (const id of Object.keys(state.companies).sort()) {
    const company = state.companies[id];
    if (!company) continue;
    for (const runId of Object.keys(company.runs).sort()) {
      const run = company.runs[runId];
      if (run?.status !== "training" || state.day < run.endDay) continue;
      if (run.held) applyCompletion(state, company, run, run.held.versionId, run.held.result);
      else run.status = "finalizing";
    }
  }
}

/** "$41,596", without the platform's locale rules. */
export function formatMoney(n: number): string {
  const digits = String(Math.abs(Math.round(n)));
  const grouped = digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return `${n < 0 ? "-" : ""}$${grouped}`;
}
