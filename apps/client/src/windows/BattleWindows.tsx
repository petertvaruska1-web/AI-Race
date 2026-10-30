import { BATTLE_SKILLS, type BattleSkill, type BattleView, type WorldView } from "@airace/protocol";
import { getBalance } from "@airace/sim";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Button } from "../components/Button.tsx";
import { INPUT } from "../components/Field.tsx";
import { dayLabel, formatMoney } from "../format.ts";
import { rivalAis } from "../rivals.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, WindowFrame, type WindowProps } from "./WindowFrame.tsx";

export const SKILL_NAMES: Record<BattleSkill, string> = {
  knowledge: "Knowledge",
  coding: "Coding",
  instruction: "Following instructions",
  pattern: "Patterns",
  creativity: "Creativity",
};

/** Why a skill can't be battled right now against `rivalProductId`, or null. */
function blocked(view: WorldView, rivalProductId: string, skill: BattleSkill): string | null {
  const b = getBalance(view.world.balance).economy.battles;
  const company = view.me.company;
  const rival = view.companies.find((c) => c.products.some((p) => p.id === rivalProductId));
  if (!company || !rival) return "Pick a rival's AI.";
  if (company.battles.some((x) => x.status === "running"))
    return "Your last battle is still going.";
  const last = company.battles.filter((x) => x.rivalId === rival.id).at(-1);
  if (last && view.world.day < last.day + b.cooldownDays) {
    return `You battled ${rival.name} recently. Try again on ${dayLabel(last.day + b.cooldownDays)}.`;
  }
  if (skill === "creativity") {
    const theirs = rival.products.find((p) => p.id === rivalProductId)?.version?.scores?.creativity;
    const mine = company.products
      .filter((p) => p.status === "live")
      .map((p) => company.versions.find((v) => v.id === p.versionId)?.scores?.creativity);
    if (typeof theirs !== "number" || !mine.some((m) => typeof m === "number")) {
      return "Creativity can't be judged on this computer yet.";
    }
  }
  return null;
}

/** Pick a rival's AI and a skill, then start the battle (game-layer design §10). */
export function NewBattleWindow({
  entry,
  view,
  rivalProductId,
}: WindowProps & { rivalProductId?: string }) {
  const socket = useSocket();
  const back = useWindows((s) => s.back);
  const targets = rivalAis(view);
  const [target, setTarget] = useState(rivalProductId ?? targets[0]?.productId ?? "");
  const [skill, setSkill] = useState<BattleSkill>("knowledge");
  const [problem, setProblem] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const fee = getBalance(view.world.balance).economy.battles.fee;
  const reason = target ? blocked(view, target, skill) : "Pick a rival's AI.";

  async function start() {
    setProblem(null);
    setSending(true);
    const res = await socket.command(view.world.id, {
      type: "startBattle",
      rivalProductId: target,
      skill,
    });
    setSending(false);
    if (res.ok) back();
    else setProblem(res.reason);
  }

  return (
    <WindowFrame
      entry={entry}
      title="Start a battle"
      subtitle={`Both AIs answer the same fresh questions. A battle costs ${formatMoney(fee)}.`}
    >
      {targets.length === 0 ? (
        <p className="text-sm text-muted">No rival has an AI on sale yet.</p>
      ) : (
        <div className="space-y-4">
          <label className="block text-sm">
            <span className="mb-1 block font-semibold">Against</span>
            <select className={INPUT} value={target} onChange={(e) => setTarget(e.target.value)}>
              {targets.map((t) => (
                <option key={t.productId} value={t.productId}>
                  {`${t.companyName}'s ${t.productName}`}
                </option>
              ))}
            </select>
          </label>
          <fieldset>
            <legend className="mb-1 text-sm font-semibold">Skill</legend>
            <div className="grid gap-2 sm:grid-cols-2">
              {BATTLE_SKILLS.map((k) => {
                const why = blocked(view, target, k);
                return (
                  <label
                    key={k}
                    className={`flex items-start gap-2 rounded-lg border border-border p-2 text-sm ${skill === k ? "bg-brand-soft" : ""}`}
                  >
                    <input
                      type="radio"
                      name="skill"
                      checked={skill === k}
                      onChange={() => setSkill(k)}
                    />
                    <span>
                      {SKILL_NAMES[k]}
                      {why && <span className="block text-xs text-muted">{why}</span>}
                    </span>
                  </label>
                );
              })}
            </div>
          </fieldset>
          <Button disabled={sending || !target || reason !== null} onClick={() => void start()}>
            Start the battle
          </Button>
          {problem && (
            <p role="alert" className="text-sm text-danger">
              {problem}
            </p>
          )}
        </div>
      )}
    </WindowFrame>
  );
}

type Transcript = { mine: string; theirs: string }[];
type StreamData =
  | { kind: "text"; round: number; side: "mine" | "theirs"; text: string }
  | { kind: "done" }
  | { kind: "calledOff" };

function wins(b: BattleView): [number, number] {
  let w = 0;
  let l = 0;
  for (const r of b.rounds ?? []) {
    if (r.mine.score > r.theirs.score) w += 1;
    if (r.mine.score < r.theirs.score) l += 1;
  }
  return [w, l];
}

/** One battle: both AIs' replies side by side, streaming live, then the result. */
export function BattleWindow({ entry, view, battleId }: WindowProps & { battleId: string }) {
  const socket = useSocket();
  const battle = view.me.company?.battles.find((b) => b.id === battleId);
  const me = view.companies.find((c) => c.id === view.me.company?.id);
  const rival = view.companies.find((c) => c.id === battle?.rivalId);
  const running = battle?.status === "running";
  const [live, setLive] = useState<Transcript>([]);
  const worldId = view.world.id;

  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    socket
      .query<{ rounds: Transcript }>({ kind: "battle", worldId, battleId })
      .then((r) => {
        if (!cancelled) setLive((now) => (now.length > 0 ? now : r.rounds));
      })
      .catch(() => {});
    const off = socket.subscribe(`battle:${worldId}:${battleId}`, (raw) => {
      const data = raw as StreamData;
      if (data.kind !== "text") return;
      setLive((now) => {
        const next = now.map((r) => ({ ...r }));
        while (next.length <= data.round) next.push({ mine: "", theirs: "" });
        const round = next[data.round] as { mine: string; theirs: string };
        round[data.side] += data.text;
        return next;
      });
    });
    return () => {
      cancelled = true;
      off();
    };
  }, [socket, worldId, battleId, running]);

  if (!battle || !me || !rival) return <GoneWindow entry={entry} />;
  const [w, l] = wins(battle);
  const outcome = battle.outcome
    ? `${battle.outcome === "win" ? "Win" : battle.outcome === "loss" ? "Loss" : "Draw"} ${w}–${l}`
    : null;
  const rounds = battle.rounds
    ? battle.rounds.map((r, round) => ({
        round,
        prompt: r.prompt,
        mine: r.mine.reply,
        theirs: r.theirs.reply,
        scores: [r.mine.score, r.theirs.score] as const,
      }))
    : live.map((r, round) => ({
        round,
        prompt: null,
        mine: r.mine,
        theirs: r.theirs,
        scores: null,
      }));
  return (
    <WindowFrame
      entry={entry}
      title={`${SKILL_NAMES[battle.skill]} battle: ${battle.mine.productName} vs ${battle.theirs.productName}`}
      subtitle={`${dayLabel(battle.day)} · against ${rival.name}`}
      wide
    >
      {outcome && (
        <motion.p
          initial={{ scale: 1.6, opacity: 0, rotate: -8 }}
          animate={{ scale: 1, opacity: 1, rotate: -4 }}
          className={`mb-4 inline-block rounded-xl border-4 px-4 py-1 font-display text-3xl font-black ${battle.outcome === "win" ? "border-good text-good" : battle.outcome === "loss" ? "border-danger text-danger" : "border-muted text-muted"}`}
        >
          {outcome}
        </motion.p>
      )}
      {battle.pointsChange !== null && (
        <p className="mb-4 text-sm font-semibold">
          {`${battle.pointsChange >= 0 ? "+" : ""}${battle.pointsChange} ranking points`}
        </p>
      )}
      {battle.status === "calledOff" && (
        <p className="mb-4 text-sm text-muted">
          This battle was called off, and the fee was refunded.
        </p>
      )}
      <div className="grid grid-cols-2 gap-4 text-xs font-semibold uppercase tracking-wider">
        <p style={{ color: me.palette.primary }}>{`${me.name}'s ${battle.mine.productName}`}</p>
        <p style={{ color: rival.palette.primary }}>
          {`${rival.name}'s ${battle.theirs.productName}`}
        </p>
      </div>
      {running && rounds.length === 0 && (
        <p className="mt-3 text-sm text-muted">Waiting for the first answers…</p>
      )}
      <ol className="mt-3 space-y-4">
        {rounds.map((r) => (
          <li key={r.round} className="rounded-xl border border-border p-3">
            {r.prompt && <p className="mb-2 text-sm font-semibold">{r.prompt}</p>}
            <div className="grid grid-cols-2 gap-4 text-sm">
              {(["mine", "theirs"] as const).map((side, i) => (
                <div key={side}>
                  <p
                    data-testid={`round-${r.round}-${side}`}
                    className="whitespace-pre-wrap break-words"
                  >
                    {r[side]}
                  </p>
                  {r.scores && (
                    <div className="mt-2 h-1.5 rounded-full bg-surface-2">
                      <motion.div
                        className="h-1.5 rounded-full"
                        style={{
                          backgroundColor: i === 0 ? me.palette.primary : rival.palette.primary,
                        }}
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.round(100 * (r.scores[i] ?? 0))}%` }}
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </li>
        ))}
      </ol>
    </WindowFrame>
  );
}
