import { Command, TEXT_LIMITS } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { campaignCommand, campaignsDay } from "./campaigns.ts";
import { computeCommand, computeDay, dataCentersDay, newCompute } from "./compute.ts";
import { fameDay } from "./fame.ts";
import { applyFeedback, newsDay } from "./feed.ts";
import { historyDay } from "./history.ts";
import { bookTraining, closeMonths, ledgerDay, newLedger } from "./ledger.ts";
import { marketDay } from "./market.ts";
import { paletteProblem } from "./palette.ts";
import { newCompanyBusiness, productCommand, productsDay } from "./products.ts";
import { rescueCommand, rescueDay } from "./rescue.ts";
import { joinRivals } from "./rivals.ts";
import { rngInt } from "./rng.ts";
import type { Input, StepResult, SystemEvent, WorldState } from "./state.ts";
import {
  activeRun,
  applyCompletion,
  findRun,
  findVersion,
  formatMoney,
  previewTraining,
  settleRuns,
  versionIdTaken,
} from "./training.ts";
import { effectiveHost } from "./world.ts";

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

/**
 * Applies one input. Pure: `state` is never changed; an accepted input returns
 * a new state with `seq` incremented, a rejected one a plain-language reason.
 */
export function step(state: WorldState, input: Input): StepResult {
  const next = structuredClone(state);
  const r = apply(next, input);
  if (!r.ok) return r;
  next.seq += 1;
  return { ok: true, state: next };
}

// Handlers mutate the fresh copy made by `step`.
function apply(s: WorldState, input: Input): Handled {
  switch (input.kind) {
    case "tick":
      return tick(s);
    case "system":
      return system(s, input.event);
    case "command":
      return command(s, input.userId, input.cmd, input.newId);
    default:
      return no("Unknown input.");
  }
}

function tick(s: WorldState): Handled {
  if (s.phase !== "running" || s.paused) return no("The world is not running.");
  s.day += 1;
  closeMonths(s);
  settleRuns(s);
  campaignsDay(s);
  productsDay(s);
  marketDay(s);
  dataCentersDay(s);
  computeDay(s);
  ledgerDay(s);
  rescueDay(s);
  newsDay(s);
  fameDay(s);
  historyDay(s);
  return OK;
}

function system(s: WorldState, e: SystemEvent): Handled {
  switch (e.type) {
    case "memberJoined": {
      if (s.members[e.userId]) return no("Already a member of this world.");
      if (Object.keys(s.members).length >= getBalance(s.balance).maxMembers)
        return no("This world is full.");
      s.members[e.userId] = {
        userId: e.userId,
        displayName: e.displayName,
        joinedSeq: s.seq + 1,
        online: true,
        companyId: null,
      };
      return OK;
    }
    case "presence": {
      const m = s.members[e.userId];
      if (!m) return no("Not a member of this world.");
      m.online = e.online;
      return OK;
    }
    case "computeStarted": {
      const found = findRun(s, e.runId);
      if (!found) return no("Unknown run.");
      if (found.run.status === "completed" || found.run.status === "failed") {
        return no("This run has finished.");
      }
      found.run.computeStarted = true;
      return OK;
    }
    case "trainingCompleted": {
      const found = findRun(s, e.runId);
      if (!found) return no("Unknown run.");
      const { company, run } = found;
      if (run.status === "completed") return no("This result was already applied.");
      if (run.status === "failed") return no("This run has finished.");
      if (run.held) return no("This result was already received.");
      if (versionIdTaken(s, e.versionId)) return no("That version id is already used.");
      // Results reveal at game pace (§7.3): an early one waits for the run's last day.
      if (run.status === "training" && s.day < run.endDay) {
        run.held = { versionId: e.versionId, result: e.result };
      } else {
        applyCompletion(s, company, run, e.versionId, e.result);
      }
      return OK;
    }
    case "trainingFailed": {
      const found = findRun(s, e.runId);
      if (!found) return no("Unknown run.");
      const { company, run } = found;
      if (run.status === "completed" || run.status === "failed")
        return no("This run has finished.");
      if (run.held) return no("This run already has its result.");
      bookTraining(s, company, run, -1);
      run.status = "failed";
      run.failure = e.reason;
      return OK;
    }
    case "evalCompleted": {
      const version = findVersion(s, e.versionId);
      if (!version) return no("Unknown version.");
      version.scores = { ...e.scores };
      version.overall = e.overall;
      version.personalityWords = [...e.personalityWords];
      version.traits = e.traits ? { ...e.traits } : null;
      return OK;
    }
    case "feedbackScored":
      return applyFeedback(s, e);
    case "rivalsJoined":
      return joinRivals(s, e.pack);
    default:
      return no("This event isn't supported yet.");
  }
}

function command(s: WorldState, userId: string, raw: unknown, newId: string | undefined): Handled {
  const member = s.members[userId];
  if (!member) return no("Not a member of this world.");
  // Re-validated here, so replayed or hostile input is checked like live input.
  const parsed = Command.safeParse(raw);
  if (!parsed.success) {
    const field = parsed.error.issues[0]?.path[0];
    if (field === "name") {
      const isProduct = (raw as { type?: unknown } | null)?.type === "launchProduct";
      const [min, max] = isProduct ? TEXT_LIMITS.productName : TEXT_LIMITS.companyName;
      return no(`${isProduct ? "Product" : "Company"} names are ${min}–${max} characters.`);
    }
    return no("That request isn't valid.");
  }
  const cmd = parsed.data;
  const isHost = effectiveHost(s) === userId;

  switch (cmd.type) {
    case "createCompany": {
      if (!newId) return no("Missing company id.");
      if (s.companies[newId]) return no("That company id is already used.");
      if (member.companyId) return no("You already have a company.");
      const lower = cmd.name.toLowerCase();
      if (Object.values(s.companies).some((c) => c.name.toLowerCase() === lower)) {
        return no("Another company already uses that name.");
      }
      const problem = paletteProblem(cmd.palette);
      if (problem) return no(problem);
      s.companies[newId] = {
        id: newId,
        ownerUserId: userId,
        startCash: getBalance(s.balance).foundingCash,
        rival: null,
        name: cmd.name,
        logo: cmd.logo,
        palette: cmd.palette,
        cash: getBalance(s.balance).foundingCash,
        runs: {},
        versions: {},
        versionSeq: 0,
        ...newCompanyBusiness(getBalance(s.balance)),
        ledger: newLedger(s.day),
        history: [],
        campaigns: {},
        compute: newCompute(getBalance(s.balance)),
        dataCenters: {},
        reports: [],
        rescue: null,
        loan: null,
        ownership: 1,
      };
      member.companyId = newId;
      return OK;
    }
    case "startWorld": {
      if (!isHost) return no("Only the host can start the world.");
      if (s.phase !== "setup") return no("The world has already started.");
      if (!member.companyId) return no("Found your company first.");
      s.phase = "running";
      return OK;
    }
    case "setSpeed": {
      if (!isHost) return no("Only the host can change the speed.");
      s.speed = cmd.speed;
      return OK;
    }
    case "setPaused": {
      if (!isHost) return no("Only the host can pause the world.");
      s.paused = cmd.paused;
      return OK;
    }
    case "startTraining": {
      if (s.phase !== "running") return no("The world is not running yet.");
      const company = member.companyId ? s.companies[member.companyId] : undefined;
      if (!company) return no("Found your company first.");
      if (!newId) return no("Missing run id.");
      if (findRun(s, newId)) return no("That run id is already used.");
      if (activeRun(company)) return no("Your AI is already training. Wait for it to finish.");
      const preview = previewTraining(s, company.id, cmd.plan);
      if (!preview.ok) return preview;
      const { cost, durationDays } = preview.preview;
      if (company.cash < cost) {
        return no(
          `Not enough cash: this training costs ${formatMoney(cost)} and you have ${formatMoney(company.cash)}.`,
        );
      }
      const seed = rngInt(s.rng, 2 ** 31);
      s.rng = seed.next;
      bookTraining(s, company, { plan: cmd.plan, preview: preview.preview }, 1);
      company.runs[newId] = {
        id: newId,
        plan: cmd.plan,
        preview: preview.preview,
        seed: seed.value,
        status: "training",
        computeStarted: false,
        startDay: s.day,
        endDay: s.day + durationDays,
        held: null,
        failure: null,
        versionId: null,
      };
      return OK;
    }
    case "publishVersion":
      return no("Sharing isn't available yet.");
    case "launchProduct":
    case "setPlan":
    case "setPrice":
    case "retireProduct": {
      const company = member.companyId ? s.companies[member.companyId] : undefined;
      if (!company) return no("Found your company first.");
      return productCommand(s, company, cmd, newId);
    }
    case "startCampaign":
    case "stopCampaign": {
      const company = member.companyId ? s.companies[member.companyId] : undefined;
      if (!company) return no("Found your company first.");
      return campaignCommand(s, company, cmd, newId);
    }
    case "buildDataCenter":
    case "setTrainingShare": {
      const company = member.companyId ? s.companies[member.companyId] : undefined;
      if (!company) return no("Found your company first.");
      return computeCommand(s, company, cmd, newId);
    }
    case "chooseRescue": {
      const company = member.companyId ? s.companies[member.companyId] : undefined;
      if (!company) return no("Found your company first.");
      return rescueCommand(s, company, cmd);
    }
    default:
      return no("That request isn't valid.");
  }
}
