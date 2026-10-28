import { GROUP_IDS, type GroupId, REGION_IDS, REGION_NAMES, type RegionId } from "@airace/protocol";
import { getBalance } from "@airace/sim";
import { useState } from "react";
import { Button } from "../components/Button.tsx";
import { INPUT } from "../components/Field.tsx";
import { formatCount, formatMoney } from "../format.ts";
import { GROUPS } from "../groups.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { WindowFrame, type WindowProps } from "./WindowFrame.tsx";

const PRESETS = [500, 2000, 10_000];

/** A new ad campaign: where, who, how much a day and for how long, with what it buys. */
export function CampaignWindow({ entry, view, region }: WindowProps & { region?: RegionId }) {
  const socket = useSocket();
  const back = useWindows((s) => s.back);
  const c = getBalance(view.world.balance).economy.campaigns;
  const [where, setWhere] = useState<RegionId | "all">(region ?? "all");
  const [who, setWho] = useState<GroupId | "all">("all");
  const [budget, setBudget] = useState(PRESETS[0] ?? c.budget.min);
  const [days, setDays] = useState(14);
  const [problem, setProblem] = useState<string | null>(null);
  const [sending, setSending] = useState(false);

  async function start() {
    setProblem(null);
    setSending(true);
    const r = await socket.command(view.world.id, {
      type: "startCampaign",
      region: where,
      group: who,
      budgetPerDay: budget,
      days,
    });
    setSending(false);
    if (r.ok) back();
    else setProblem(r.reason);
  }

  return (
    <WindowFrame
      entry={entry}
      title="New ad campaign"
      subtitle="Ads make more people aware of your company, a day at a time."
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="block text-sm">
          <span className="mb-1 block text-xs font-semibold uppercase tracking-wider text-muted">
            Where
          </span>
          <select
            aria-label="Where"
            className={INPUT}
            value={where}
            onChange={(e) => setWhere(e.target.value as RegionId | "all")}
          >
            <option value="all">Everywhere</option>
            {REGION_IDS.map((r) => (
              <option key={r} value={r}>
                {REGION_NAMES[r]}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-sm">
          <span className="mb-1 block text-xs font-semibold uppercase tracking-wider text-muted">
            Who
          </span>
          <select
            aria-label="Who"
            className={INPUT}
            value={who}
            onChange={(e) => setWho(e.target.value as GroupId | "all")}
          >
            <option value="all">Everyone</option>
            {GROUP_IDS.map((g) => (
              <option key={g} value={g}>
                {GROUPS[g].many}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="mt-5">
        <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-muted">
          Budget a day
        </p>
        <div className="flex flex-wrap items-center gap-2">
          {PRESETS.map((p) => (
            <button
              key={p}
              type="button"
              aria-pressed={budget === p}
              onClick={() => setBudget(p)}
              className={`rounded-md px-3 py-1 text-sm transition ${budget === p ? "bg-brand text-on-brand" : "bg-surface-2 text-muted hover:text-text"}`}
            >
              {formatMoney(p)}
            </button>
          ))}
          <input
            type="range"
            aria-label="Budget a day"
            className="ml-2 flex-1 accent-[var(--brand)]"
            min={c.budget.min}
            max={c.budget.max}
            step={100}
            value={budget}
            onChange={(e) => setBudget(Number(e.target.value))}
          />
          <span className="tabular w-24 text-right font-semibold">{formatMoney(budget)}</span>
        </div>
      </div>

      <label className="mt-5 block">
        <span className="mb-1 block text-xs font-semibold uppercase tracking-wider text-muted">
          How long
        </span>
        <span className="flex items-center gap-3">
          <input
            type="range"
            aria-label="Days"
            className="flex-1 accent-[var(--brand)]"
            min={c.days.min}
            max={c.days.max}
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
          />
          <span className="tabular w-24 text-right font-semibold">{days} game days</span>
        </span>
      </label>

      <p className="mt-5 rounded-xl border border-border bg-surface-2/60 p-3 text-sm">
        Reaches about {formatCount(Math.round(budget / c.costPerReach))} people a day ·{" "}
        {formatMoney(budget * days)} in total
      </p>
      {problem && (
        <p role="alert" className="mt-3 text-sm text-danger">
          {problem}
        </p>
      )}
      <div className="mt-5 flex justify-end gap-3">
        <Button variant="secondary" onClick={back}>
          Cancel
        </Button>
        <Button onClick={() => void start()} disabled={sending}>
          Start the campaign
        </Button>
      </div>
    </WindowFrame>
  );
}
