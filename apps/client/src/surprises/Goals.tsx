import {
  type Goal,
  type GoalSpec,
  MILESTONE_NAMES,
  type NewsView,
  type WorldView,
} from "@airace/protocol";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Button } from "../components/Button.tsx";
import { DayChart } from "../components/charts/DayChart.tsx";
import { MonthBars } from "../components/charts/MonthBars.tsx";
import { INPUT } from "../components/Field.tsx";
import { dayLabel, formatCount, formatMoney } from "../format.ts";
import { daysOf } from "../history.ts";
import { useWindows } from "../store/windows.ts";
import { Section, WindowFrame, type WindowProps } from "../windows/WindowFrame.tsx";
import { MILESTONE_ICONS } from "./Trophies.tsx";

type Kind = Goal["kind"];
const RIVAL_KINDS: Kind[] = ["bestAi", "beatArch"];
const DEFAULT_TARGETS: Partial<Record<Kind, number>> = {
  customers: 1_000_000,
  profit: 1_000_000,
  value: 100_000_000,
};
const DEADLINES = [
  { label: "No deadline", years: null },
  { label: "1 year", years: 1 },
  { label: "2 years", years: 2 },
  { label: "5 years", years: 5 },
] as const;

/** A goal in the words the picker and the progress bar use. */
export function goalTitle(kind: Kind, target: number | null): string {
  switch (kind) {
    case "customers":
      return `Reach ${formatCount(target ?? 0)} customers`;
    case "profit":
      return `Make ${formatMoney(target ?? 0)} profit in a month`;
    case "bestAi":
      return "Build the best AI overall";
    case "beatArch":
      return "Beat the arch-rival on customers";
    case "value":
      return `Grow the company's value to ${formatMoney(target ?? 0)}`;
    default:
      return "No goal: play as long as you like";
  }
}

function specOf(kind: Kind, target: number, years: 1 | 2 | 5 | null): GoalSpec {
  const goal = (
    kind === "customers" || kind === "profit" || kind === "value" ? { kind, target } : { kind }
  ) as Goal;
  return { goal, deadlineYears: kind === "none" ? null : years };
}

/**
 * The goal picker (game-layer design §11) in "Start a new world": cards for each goal, an
 * editable target where it has one, and a deadline. Rival goals hide without the rival AIs.
 */
export function GoalPicker({
  rivals,
  onChange,
}: {
  rivals: boolean;
  onChange: (spec: GoalSpec) => void;
}) {
  const kinds: Kind[] = ["none", "customers", "profit", "bestAi", "beatArch", "value"];
  const shown = kinds.filter((k) => rivals || !RIVAL_KINDS.includes(k));
  const [kind, setKind] = useState<Kind>("none");
  const [targets, setTargets] = useState(DEFAULT_TARGETS);
  const [years, setYears] = useState<1 | 2 | 5 | null>(null);
  const target = targets[kind] ?? 0;

  useEffect(() => {
    onChange(specOf(kind, target, years));
  }, [kind, target, years, onChange]);

  return (
    <fieldset className="space-y-3">
      <legend className="mb-2 text-sm font-semibold">A goal (optional)</legend>
      <div className="grid gap-2 sm:grid-cols-2">
        {shown.map((k) => (
          <label
            key={k}
            className={`cursor-pointer rounded-xl border p-3 text-sm ${k === kind ? "border-brand bg-brand-soft" : "border-border"}`}
          >
            <input
              type="radio"
              name="goal"
              className="sr-only"
              checked={k === kind}
              onChange={() => setKind(k)}
            />
            {goalTitle(k, targets[k] ?? null)}
          </label>
        ))}
      </div>
      {DEFAULT_TARGETS[kind] !== undefined && (
        <label className="block text-sm">
          <span className="mb-1 block font-semibold">Target</span>
          <input
            type="number"
            className={INPUT}
            aria-label="Goal target"
            min={kind === "value" ? 100_000 : 1_000}
            value={target}
            onChange={(e) =>
              setTargets((t) => ({ ...t, [kind]: Math.round(Number(e.target.value)) }))
            }
          />
        </label>
      )}
      {kind !== "none" && (
        <label className="block text-sm">
          <span className="mb-1 block font-semibold">Deadline</span>
          <select
            className={INPUT}
            aria-label="Goal deadline"
            value={years ?? ""}
            onChange={(e) =>
              setYears(e.target.value ? (Number(e.target.value) as 1 | 2 | 5) : null)
            }
          >
            {DEADLINES.map((d) => (
              <option key={d.label} value={d.years ?? ""}>
                {d.label}
              </option>
            ))}
          </select>
        </label>
      )}
    </fieldset>
  );
}

function progressText(kind: Kind, value: number, target: number | null): string {
  const money = kind === "profit" || kind === "value";
  const show = (n: number) => (money ? formatMoney(n) : formatCount(n));
  if (kind === "bestAi") return value > 0 ? `#${value} overall` : "Not ranked yet";
  return target === null ? show(value) : `${show(value)} / ${show(target)}`;
}

/** The goal's progress along the bottom of the Company screen; it opens the goal window. */
export function GoalProgress({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const goal = view.world.goal;
  const progress = view.me.company?.goalProgress;
  if (!goal || !progress) return null;
  const missed = goal.missedDay !== null;
  if (missed) {
    return (
      <div className="window px-4 py-3 text-sm" data-testid="goal-missed">
        <p className="font-semibold">{goalTitle(goal.kind, goal.target)}</p>
        <p className="text-muted">
          The deadline passed. Your company carries on: play as long as you like.
        </p>
      </div>
    );
  }
  const share =
    goal.kind === "bestAi"
      ? progress.done
        ? 1
        : 0
      : progress.target
        ? Math.min(1, Math.max(0, progress.value / progress.target))
        : 0;
  const left = goal.deadlineDay === null ? null : goal.deadlineDay - view.world.day;
  return (
    <button
      type="button"
      onClick={(e) => open({ kind: "goal" }, e.currentTarget)}
      className="window w-full px-4 py-3 text-left"
      data-testid="goal-progress"
    >
      <span className="flex items-baseline justify-between gap-3 text-sm">
        <span className="font-semibold">{goalTitle(goal.kind, goal.target)}</span>
        <span className="tabular text-muted">
          {progressText(goal.kind, progress.value, progress.target)}
        </span>
      </span>
      <span className="mt-2 block h-2 rounded-full bg-surface-2">
        <motion.span
          className="block h-2 rounded-full bg-brand"
          initial={{ width: 0 }}
          animate={{ width: `${100 * share}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </span>
      <span className="mt-1 block text-xs text-muted">
        {goal.reachedDay !== null
          ? `Reached on ${dayLabel(goal.reachedDay)}`
          : RIVAL_KINDS.includes(goal.kind) && view.world.rivals === "none"
            ? "Rivals haven't joined yet"
            : left === null
              ? "No deadline"
              : `${formatCount(Math.max(0, left))} days left`}
      </span>
    </button>
  );
}

/** The goal and how the company has moved towards it, day by day. */
export function GoalWindow({ entry, view }: WindowProps) {
  const goal = view.world.goal;
  const company = view.me.company;
  if (!goal || !company) return null;
  const days = daysOf(company);
  const series =
    goal.kind === "value" || goal.kind === "profit"
      ? { id: "cash", label: "Cash", values: days.map((d) => d.cash), format: formatMoney }
      : {
          id: "customers",
          label: "Customers",
          values: days.map((d) => Object.values(d.customers).reduce((a, b) => a + b, 0)),
          format: formatCount,
        };
  return (
    <WindowFrame entry={entry} title={goalTitle(goal.kind, goal.target)} wide>
      <DayChart
        label={series.label}
        days={days.map((d) => d.day)}
        series={[
          { id: series.id, label: series.label, color: "var(--brand)", values: series.values },
        ]}
        format={series.format}
      />
      {goal.deadlineDay !== null && (
        <p className="mt-3 text-sm text-muted">{`Deadline: ${dayLabel(goal.deadlineDay)}`}</p>
      )}
    </WindowFrame>
  );
}

const SEEN_KEY = (worldId: string) => `airace.victory.${worldId}`;

function seenVictory(worldId: string): boolean {
  try {
    return localStorage.getItem(SEEN_KEY(worldId)) === "1";
  } catch {
    return false;
  }
}

function markVictory(worldId: string): void {
  try {
    localStorage.setItem(SEEN_KEY(worldId), "1");
  } catch {
    // Without storage it shows again next time; nothing else depends on it.
  }
}

/**
 * The victory screen (game-layer design §11), once per world in this browser, when the
 * player's company reached the goal: the company's story, drawn in, then "Keep playing".
 */
export function Victory({ view }: { view: WorldView }) {
  const goal = view.world.goal;
  const company = view.me.company;
  const won = !!goal && goal.reachedDay !== null && goal.reachedBy === company?.id;
  const [show, setShow] = useState(false);
  useEffect(() => {
    if (won && !seenVictory(view.world.id)) setShow(true);
  }, [won, view.world.id]);
  if (!show || !goal || !company) return null;

  const close = () => {
    markVictory(view.world.id);
    setShow(false);
  };
  const days = daysOf(company);
  const name = view.companies.find((c) => c.id === company.id)?.name ?? "Your company";
  const reports = company.reports;
  const best = reports.map((r) => r.best).find((b) => b !== null) ?? null;
  const won_battles = company.battles.filter((b) => b.outcome === "win").length;
  const mine = view.feed.filter(
    (f): f is NewsView => f.kind === "news" && f.companyId === company.id,
  );
  const biggest = mine.reduce<NewsView | null>(
    (a, b) => (a === null || Math.abs(b.sentiment) > Math.abs(a.sentiment) ? b : a),
    null,
  );
  const step = (k: number) => ({
    initial: { opacity: 0, y: 16 },
    animate: { opacity: 1, y: 0 },
    transition: { delay: 0.3 + 0.35 * k, duration: 0.5 },
  });

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="You reached your goal"
      className="fixed inset-0 z-50 overflow-y-auto bg-bg/90 p-6 backdrop-blur"
    >
      <div className="mx-auto max-w-3xl space-y-5">
        <motion.h1
          className="font-display text-4xl font-bold text-brand"
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 200, damping: 16 }}
        >
          {`${name} did it!`}
        </motion.h1>
        <motion.p {...step(0)} className="text-lg">
          {`${goalTitle(goal.kind, goal.target)}: reached on ${dayLabel(goal.reachedDay ?? 0)}.`}
        </motion.p>
        <motion.div {...step(1)} className="window p-4">
          <DayChart
            label="Customers"
            days={days.map((d) => d.day)}
            series={[
              {
                id: "customers",
                label: "Customers",
                color: "var(--brand)",
                values: days.map((d) => Object.values(d.customers).reduce((a, b) => a + b, 0)),
              },
              {
                id: "cash",
                label: "Cash",
                color: "var(--brand-2)",
                values: days.map((d) => d.cash),
              },
            ]}
            format={formatCount}
            height={180}
          />
        </motion.div>
        {reports.length > 0 && (
          <motion.div {...step(2)} className="window p-4">
            <MonthBars
              label="Profit by month"
              bars={reports.map((r) => ({ month: r.month, value: r.profit }))}
              format={formatMoney}
              height={120}
            />
          </motion.div>
        )}
        {company.milestones.length > 0 && (
          <motion.div {...step(3)}>
            <Section title="Milestones">
              <ul className="flex flex-wrap gap-3 text-sm">
                {company.milestones.map((m) => (
                  <li key={m.id}>
                    <span aria-hidden="true">{MILESTONE_ICONS[m.id]}</span> {MILESTONE_NAMES[m.id]}
                    <span className="text-xs text-muted">{` · ${dayLabel(m.day)}`}</span>
                  </li>
                ))}
              </ul>
            </Section>
          </motion.div>
        )}
        {best && (
          <motion.div {...step(4)}>
            <Section title="A moment customers loved">
              <div className="rounded-lg bg-surface-2 p-3 text-sm">
                <p className="player-text text-muted">{best.question}</p>
                <p className="player-text mt-1">{best.reply}</p>
              </div>
            </Section>
          </motion.div>
        )}
        <motion.div {...step(5)} className="flex flex-wrap gap-6 text-sm">
          {won_battles > 0 && <p>{`Battles won: ${won_battles}`}</p>}
          {biggest && <p className="player-text">{`Biggest headline: ${biggest.headline}`}</p>}
        </motion.div>
        <motion.div {...step(6)}>
          <Button onClick={close}>Keep playing</Button>
        </motion.div>
      </div>
    </div>
  );
}
