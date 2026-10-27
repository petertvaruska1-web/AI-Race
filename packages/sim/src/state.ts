import type {
  Command,
  GroupId,
  Logo,
  Palette,
  PlanId,
  RegionId,
  RunStatus,
  Speed,
  TrainingPlan,
  TrainingPreview,
} from "@airace/protocol";
import type { BalanceVersion } from "./balance.ts";
import type { RngState } from "./rng.ts";

export type Member = {
  userId: string;
  displayName: string;
  /** The `seq` of the input that added the member; orders the host handover. */
  joinedSeq: number;
  online: boolean;
  companyId: string | null;
};

/** What a worker reports when a training job ends. */
export type TrainingResult = {
  steps: number;
  finalLoss: number | null;
  heldoutLosses: Record<string, number | null>;
  unstable: boolean;
};

export type RunState = {
  id: string;
  plan: TrainingPlan;
  preview: TrainingPreview;
  /** Drawn from the world rng, so a replayed run trains the same model. */
  seed: number;
  status: RunStatus;
  computeStarted: boolean;
  startDay: number;
  endDay: number;
  /** A result that arrived before its game time; applied by the tick that reaches `endDay`. */
  held: { versionId: string; result: TrainingResult } | null;
  failure: string | null;
  versionId: string | null;
};

export type VersionState = {
  id: string;
  label: string;
  runId: string;
  parentId: string | null;
  nLayer: number;
  dModel: number;
  ctxLen: number;
  createdDay: number;
  trainedTokens: number;
  scores: Record<string, number | null> | null;
  overall: number | null;
  personalityWords: string[];
  published: boolean;
};

export type ProductState = {
  id: string;
  name: string;
  versionId: string;
  status: "countdown" | "live" | "retired";
  announcedDay: number;
  launchDay: number;
  hype: number;
  /** Plans it powers from launch day. */
  launchPlans: PlanId[];
  /** Feedback batches already applied (the last 20), so a result is never counted twice. */
  feedbackKeys: string[];
};

export type PlansState = {
  free: { productId: string | null };
  personal: { productId: string | null; price: number };
  business: { productId: string | null; price: number };
};

export type Spotlight = {
  id: string;
  name: string;
  handle: string;
  group: GroupId;
  region: RegionId;
  bio: string;
};

/** Customer counts are fractional (they move smoothly); views round them. */
export type MarketState = {
  customers: Record<
    string,
    Partial<Record<string, Partial<Record<GroupId, Partial<Record<PlanId, number>>>>>>
  >;
  awareness: Record<string, Partial<Record<string, Partial<Record<GroupId, number>>>>>;
  satisfaction: Record<string, Partial<Record<GroupId, number>>>;
};

export type FameState = { reputation: number; followers: number };

export type CompanyState = {
  id: string;
  ownerUserId: string;
  name: string;
  logo: Logo;
  palette: Palette;
  cash: number;
  runs: Record<string, RunState>;
  versions: Record<string, VersionState>;
  versionSeq: number;
  products: Record<string, ProductState>;
  plans: PlansState;
  fame: FameState;
};

export type WorldState = {
  worldId: string;
  name: string;
  balance: BalanceVersion;
  rng: RngState;
  phase: "setup" | "running";
  day: number;
  speed: Speed;
  paused: boolean;
  hostUserId: string;
  members: Record<string, Member>;
  companies: Record<string, CompanyState>;
  inviteCode: string;
  /** Accepted inputs so far. */
  seq: number;
  market: MarketState;
  spotlights: Spotlight[];
};

export type SystemEvent =
  | { type: "memberJoined"; userId: string; displayName: string }
  | { type: "presence"; userId: string; online: boolean }
  | { type: "trainingCompleted"; runId: string; versionId: string; result: TrainingResult }
  | { type: "trainingFailed"; runId: string; reason: string }
  | { type: "computeStarted"; runId: string }
  | {
      type: "evalCompleted";
      versionId: string;
      scores: Record<string, number | null>;
      overall: number | null;
      personalityWords: string[];
    };

/** Everything that can change a world. Logged, so replay reproduces the state. */
export type Input =
  | { kind: "command"; userId: string; cmd: Command; newId?: string }
  | { kind: "tick" }
  | { kind: "system"; event: SystemEvent };

export type StepResult = { ok: true; state: WorldState } | { ok: false; reason: string };
