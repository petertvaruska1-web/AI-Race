import {
  BENCH_CATEGORIES,
  type Command,
  RIVAL_STYLES,
  type RivalPack,
  type RivalStyle,
} from "@airace/protocol";
import {
  type BalanceVersion,
  createWorld,
  type Input,
  type StepResult,
  step,
  type WorldState,
} from "../src/index.ts";

// A rival pack for tests: eight lineages (two per style), three versions each,
// whose scores grow per version and lean towards the lineage's speciality.

const FIRST: Record<string, number> = {
  language: 60,
  reasoning: 8,
  pattern: 80,
  knowledge: 44,
  coding: 38,
  creativity: 2,
  consistency: 37,
  instruction: 63,
};

const LINEAGES: { id: string; style: RivalStyle; focus: string; lean: string }[] = [
  { id: "premium-a", style: "premium", focus: "businesses", lean: "instruction" },
  { id: "premium-b", style: "premium", focus: "researchers", lean: "knowledge" },
  { id: "budget-a", style: "budget", focus: "everyday", lean: "language" },
  { id: "budget-b", style: "budget", focus: "students", lean: "knowledge" },
  { id: "specialist-code", style: "specialist", focus: "developers", lean: "coding" },
  { id: "specialist-stories", style: "specialist", focus: "creators", lean: "creativity" },
  { id: "consumer-a", style: "consumer", focus: "everyday", lean: "language" },
  { id: "consumer-b", style: "consumer", focus: "creators", lean: "creativity" },
];

const SHAPES = [
  [4, 128],
  [6, 192],
  [8, 256],
] as const;

function version(id: string, lean: string, premium: boolean, k: number) {
  const scores: Record<string, number | null> = {};
  for (const c of BENCH_CATEGORIES) {
    const base = (FIRST[c] ?? 0) + 6 * k + (premium ? 4 : 0) + (c === lean ? 15 : 0);
    scores[c] = Math.min(100, base);
  }
  const values = Object.values(scores) as number[];
  const shape = SHAPES[k] ?? SHAPES[2];
  return {
    key: `${id}-v${k + 1}`,
    nLayer: shape[0],
    dModel: shape[1],
    ctxLen: 256,
    trainedTokens: 12_582_912 * (k + 1),
    scores,
    overall: values.reduce((a, b) => a + b, 0) / values.length,
    traits: { verbosity: 0.3 + 0.1 * k, warmth: 0.5 },
    personalityWords: k > 0 ? ["talkative"] : [],
    breakdown: { "knowledge/topic:animals": [40 + 5 * k, 12] as [number, number] },
  };
}

export const SYNTHETIC_PACK: RivalPack = {
  version: "rivals-v1",
  dataIdentity: "synthetic",
  tokenizer: "tok-v1",
  suite: "bench-v1",
  quick: false,
  lineages: LINEAGES.map((l) => ({
    id: l.id,
    style: l.style,
    focus: l.focus as RivalPack["lineages"][number]["focus"],
    versions: [0, 1, 2].map((k) => version(l.id, l.lean, l.style === "premium", k)),
  })),
};

export const STYLES = RIVAL_STYLES;

export function ok(r: StepResult): WorldState {
  if (!r.ok) throw new Error(`rejected: ${r.reason}`);
  return r.state;
}

export function rejected(r: StepResult): string {
  if (r.ok) throw new Error("expected a rejection");
  return r.reason;
}

export const cmd = (c: unknown, newId?: string, userId = "u1"): Input => ({
  kind: "command",
  userId,
  cmd: c as Command,
  ...(newId ? { newId } : {}),
});
export const sys = (event: unknown): Input => ({ kind: "system", event }) as Input;
export const TICK: Input = { kind: "tick" };

export function ticks(s: WorldState, n: number): WorldState {
  let out = s;
  for (let i = 0; i < n; i++) out = ok(step(out, TICK));
  return out;
}

/** A running world where u1 founded "Acme" (company c1), on `balance`. */
export function runningWorld(balance: BalanceVersion = "balance-v2", seed = 7): WorldState {
  let s = createWorld({
    worldId: "w1",
    name: "Rivals",
    hostUserId: "u1",
    hostName: "Ana",
    seed,
    balance,
    inviteCode: "INV",
  });
  s = ok(
    step(
      s,
      cmd(
        {
          type: "createCompany",
          name: "Acme",
          logo: { shape: "circle", symbol: "spark", initials: "AC" },
          palette: { primary: "#ff9f1c", secondary: "#2ec4b6" },
        },
        "c1",
      ),
    ),
  );
  return ok(step(s, cmd({ type: "startWorld" })));
}

/** `runningWorld` with the synthetic pack's rivals joined. */
export function withRivals(balance: BalanceVersion = "balance-v2", seed = 7): WorldState {
  return ok(step(runningWorld(balance, seed), sys({ type: "rivalsJoined", pack: SYNTHETIC_PACK })));
}
