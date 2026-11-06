import type { TrainingPlan, TrainingPreview } from "@airace/protocol";
import { type Balance, getBalance } from "./balance.ts";
import { ownedPflops } from "./compute.ts";
import { strengthsOf } from "./departments.ts";
import { paramCount, trainingFlops } from "./shape.ts";
import type { CompanyState, RunState, TrainingResult, VersionState, WorldState } from "./state.ts";

export type PreviewResult = { ok: true; preview: TrainingPreview } | { ok: false; reason: string };

/** What a plan costs in money and game days, or why it can't run. */
/** Coaching: a plan can train on lessons from this many; a company keeps at most `LESSONS_MAX`. */
export const LESSONS_MIN = 5;
export const LESSONS_MAX = 200;

export function previewTraining(
  state: WorldState,
  companyId: string,
  plan: TrainingPlan,
): PreviewResult {
  const company = state.companies[companyId];
  if (!company) return { ok: false, reason: "Unknown company." };
  if (plan.parentVersionId !== null && !company.versions[plan.parentVersionId]) {
    return { ok: false, reason: "That AI isn't one of yours." };
  }
  const parent =
    plan.parentVersionId === null ? null : (company.versions[plan.parentVersionId] ?? null);
  const lent = ownedPflops(company, state.day) * company.compute.trainingShare;
  return previewPlan(
    getBalance(state.balance),
    parent,
    plan,
    lent,
    company.loved,
    company.lessons ?? 0,
    strengthsOf(state, company),
  );
}

/** The departments a plan's price and reading depend on (C43), each a strength 0–1. */
export type PlanTeams = { research?: number; data?: number };

/**
 * The same preview from the balance, the parent's brain and the data-center compute
 * lent to training (PFLOP a day) alone, so a client can show exactly the cost and
 * time the server will charge. Lent compute is free and runs beside the cloud.
 */
export function previewPlan(
  b: Balance,
  parent: { nLayer: number; dModel: number } | null,
  plan: TrainingPlan,
  ownedPflopsPerDay = 0,
  /** Conversations the company's customers loved (for "Learn from customers"). */
  loved = 0,
  /** Lessons the player gave the AI (for "Train on your lessons"). */
  lessons = 0,
  /** Research reads more text for the same money; the Data team makes checks cheaper. */
  teams: PlanTeams = {},
): PreviewResult {
  if (Object.values(plan.mix).every((w) => w === 0)) {
    return { ok: false, reason: "Pick at least one dataset to train on." };
  }
  const brain = b.brainSizes.find((s) => s.id === plan.brainSize);
  if (!brain) return { ok: false, reason: "Unknown brain size." };
  const length = b.lengths.find((l) => l.id === plan.length);
  if (!length) return { ok: false, reason: "Unknown training length." };
  if (!b.attentionSpans.includes(plan.attentionSpan))
    return { ok: false, reason: "Unknown attention span." };
  if (plan.review > 0 && plan.parentVersionId === null) {
    return { ok: false, reason: "Reviewing needs an earlier AI to review. Pick one to grow from." };
  }
  // Growth keeps what the parent learned, so a brain can only grow (R6).
  if (parent && (brain.nLayer < parent.nLayer || brain.dModel < parent.dModel)) {
    return {
      ok: false,
      reason: "A new version can't have a smaller brain than the AI it grows from.",
    };
  }
  const e = b.economy;
  if (plan.coaching === true && lessons < LESSONS_MIN) {
    return {
      ok: false,
      reason: `Your AI needs at least ${LESSONS_MIN} lessons first. You have ${lessons}.`,
    };
  }
  const learn = plan.learnFromCustomers === true;
  if (learn && loved < e.lovedMin) {
    return {
      ok: false,
      reason: `Customers have loved ${loved} of your AI's answers so far. Learning from them needs at least ${e.lovedMin}.`,
    };
  }
  // Tutoring packs: extra practice from one dataset on top of the length.
  const packs = plan.tutoring?.packs ?? 0;
  const paid = Math.round(length.tokens * (1 + e.tutoring.packShare * packs));
  // Research: the AI reads more text for the money and time the paid text costs.
  const tokens = Math.round(paid * (1 + e.departments.research * (teams.research ?? 0)));
  const params = paramCount(brain.nLayer, brain.dModel);
  const flops = trainingFlops(params, paid);
  const pflop = flops / 1e15;
  const listed =
    (plan.factCheck ? b.factCheckCost : 0) +
    (plan.cleaning === "expert" ? Math.round((paid / 1000) * e.expertCheckPerThousand) : 0) +
    (learn ? Math.min(loved, e.lovedMax) * e.lovedConversationFee : 0);
  // The Data team: every check costs less.
  const checks = Math.round(listed * (1 - e.departments.data * (teams.data ?? 0)));
  const compute = Math.round(
    (pflop * b.cloudPricePerPflop * b.cloudPflopsPerDay) /
      (b.cloudPflopsPerDay + ownedPflopsPerDay),
  );
  return {
    ok: true,
    preview: {
      params,
      tokens,
      flops,
      cost: compute + packs * e.tutoring.packFee + checks,
      checks,
      durationDays: Math.max(1, Math.ceil(pflop / (b.cloudPflopsPerDay + ownedPflopsPerDay))),
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

export function findVersion(state: WorldState, versionId: string): VersionState | null {
  for (const id of Object.keys(state.companies).sort()) {
    const version = state.companies[id]?.versions[versionId];
    if (version) return version;
  }
  return null;
}

/** Whether a version id is used by a version or by a result waiting for its game day. */
export function versionIdTaken(state: WorldState, versionId: string): boolean {
  if (findVersion(state, versionId)) return true;
  return Object.values(state.companies).some((c) =>
    Object.values(c.runs).some((r) => r.held?.versionId === versionId),
  );
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
    traits: null,
    published: false,
    packKey: null,
    unchecked:
      parent?.unchecked === true ||
      (run.plan.mix.web > 0 && !run.plan.factCheck && run.plan.cleaning !== "expert"),
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
