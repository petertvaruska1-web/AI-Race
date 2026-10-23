import type {
  PrivateCompany,
  PublicCompany,
  PublicVersion,
  RunView,
  TrainingPlan,
  VersionView,
  WorldView,
} from "@airace/protocol";
import { dateOfDay } from "./calendar.ts";
import type { CompanyState, RunState, VersionState, WorldState } from "./state.ts";
import { runProgress } from "./training.ts";
import { effectiveHost } from "./world.ts";

// The only path from world state to a client (R2, spec §6.4). Every view is
// built field by field: copying or filtering state objects would let a field
// added later leak by default.

/** What `viewerUserId` may see of the world. Throws for a non-member. */
export function projectView(state: WorldState, viewerUserId: string): WorldView {
  const viewer = state.members[viewerUserId];
  if (!viewer) throw new Error("not a member of this world");
  const own = viewer.companyId ? state.companies[viewer.companyId] : undefined;
  return {
    world: {
      id: state.worldId,
      name: state.name,
      phase: state.phase,
      day: state.day,
      date: dateOfDay(state.day),
      speed: state.speed,
      paused: state.paused,
      hostUserId: state.hostUserId,
      effectiveHostUserId: effectiveHost(state),
      inviteCode: state.inviteCode,
    },
    members: Object.values(state.members)
      .sort((a, b) => a.joinedSeq - b.joinedSeq)
      .map((m) => ({
        userId: m.userId,
        displayName: m.displayName,
        online: m.online,
        companyId: m.companyId,
      })),
    companies: sortedIds(state.companies).map((id) =>
      publicCompany(state.companies[id] as CompanyState),
    ),
    me: { userId: viewerUserId, company: own ? privateCompany(state, own) : null },
  };
}

function sortedIds(record: Record<string, unknown>): string[] {
  return Object.keys(record).sort();
}

function publicCompany(c: CompanyState): PublicCompany {
  return {
    id: c.id,
    name: c.name,
    logo: { shape: c.logo.shape, symbol: c.logo.symbol, initials: c.logo.initials },
    palette: { primary: c.palette.primary, secondary: c.palette.secondary },
    ownerUserId: c.ownerUserId,
    publishedVersions: versionsInOrder(c)
      .filter((v) => v.published)
      .map(publicVersion),
  };
}

function publicVersion(v: VersionState): PublicVersion {
  return {
    id: v.id,
    label: v.label,
    createdDay: v.createdDay,
    scores: v.scores ? { ...v.scores } : null,
    overall: v.overall,
    personalityWords: [...v.personalityWords],
  };
}

function privateCompany(state: WorldState, c: CompanyState): PrivateCompany {
  const runs = Object.values(c.runs).sort((a, b) => a.startDay - b.startDay || cmp(a.id, b.id));
  return {
    id: c.id,
    cash: c.cash,
    runs: runs.map((r) => runView(state, r)),
    versions: versionsInOrder(c).map(versionView),
  };
}

// A held result stays out of the view: it is revealed when its game day comes (§7.3).
function runView(state: WorldState, r: RunState): RunView {
  return {
    id: r.id,
    status: r.status,
    computeStarted: r.computeStarted,
    plan: planView(r.plan),
    preview: {
      params: r.preview.params,
      tokens: r.preview.tokens,
      flops: r.preview.flops,
      cost: r.preview.cost,
      durationDays: r.preview.durationDays,
    },
    startDay: r.startDay,
    endDay: r.endDay,
    progress: r.status === "completed" ? 1 : runProgress(r, state.day),
    failure: r.failure,
    versionId: r.versionId,
  };
}

function planView(p: TrainingPlan): TrainingPlan {
  return {
    mix: { ...p.mix },
    cleaning: p.cleaning,
    removeDuplicates: p.removeDuplicates,
    factCheck: p.factCheck,
    variety: p.variety,
    brainSize: p.brainSize,
    attentionSpan: p.attentionSpan,
    length: p.length,
    learningStyle: p.learningStyle,
    review: p.review,
    probePrompt: p.probePrompt,
    parentVersionId: p.parentVersionId,
  };
}

function versionView(v: VersionState): VersionView {
  return {
    id: v.id,
    label: v.label,
    runId: v.runId,
    parentId: v.parentId,
    shape: { nLayer: v.nLayer, dModel: v.dModel, ctxLen: v.ctxLen },
    createdDay: v.createdDay,
    trainedTokens: v.trainedTokens,
    scores: v.scores ? { ...v.scores } : null,
    overall: v.overall,
    personalityWords: [...v.personalityWords],
    published: v.published,
  };
}

/** Versions oldest first (`v1`, `v2`, …). */
function versionsInOrder(c: CompanyState): VersionState[] {
  return Object.values(c.versions).sort((a, b) => labelNumber(a) - labelNumber(b));
}

function labelNumber(v: VersionState): number {
  return Number(v.label.slice(1));
}

function cmp(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
