import { Command, TEXT_LIMITS } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { paletteProblem } from "./palette.ts";
import type { Input, StepResult, SystemEvent, WorldState } from "./state.ts";
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
      const [min, max] = TEXT_LIMITS.companyName;
      return no(`Company names are ${min}–${max} characters.`);
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
        name: cmd.name,
        logo: cmd.logo,
        palette: cmd.palette,
        cash: getBalance(s.balance).foundingCash,
        runs: {},
        versions: {},
        versionSeq: 0,
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
    case "startTraining":
      return no("Training isn't available yet.");
    case "publishVersion":
      return no("Sharing isn't available yet.");
    default:
      return no("That request isn't valid.");
  }
}
