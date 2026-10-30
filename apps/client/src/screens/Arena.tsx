import {
  BATTLE_SKILLS,
  type BattleSkill,
  BENCH_CATEGORIES,
  type LeagueTable,
  type WorldView,
} from "@airace/protocol";
import { useState } from "react";
import { Button } from "../components/Button.tsx";
import { Logo } from "../components/Logo.tsx";
import { Panel } from "../components/Panel.tsx";
import { dayLabel, formatCount, formatMoney } from "../format.ts";
import { useWindows } from "../store/windows.ts";

const TABLE_NAMES: Record<LeagueTable, string> = {
  customers: "Customers",
  revenue: "Revenue",
  overall: "Overall",
  language: "Language",
  reasoning: "Reasoning",
  pattern: "Patterns",
  knowledge: "Knowledge",
  coding: "Coding",
  creativity: "Creativity",
  consistency: "Consistency",
  instruction: "Instructions",
};
type Tab = LeagueTable | "points";
const TABS: Tab[] = ["customers", "revenue", "overall", ...BENCH_CATEGORIES, "points"];

function valueText(table: LeagueTable, value: number | null): string {
  if (value === null) return "—";
  if (table === "customers") return formatCount(value);
  if (table === "revenue") return formatMoney(value);
  return String(Math.round(value));
}

function Rows({
  view,
  label,
  rows,
}: {
  view: WorldView;
  label: string;
  rows: { companyId: string; rank: number; text: string }[];
}) {
  const open = useWindows((s) => s.open);
  const me = view.me.company?.id;
  return (
    <table aria-label={label} className="w-full text-sm">
      <thead>
        <tr className="text-left text-xs uppercase tracking-wider text-muted">
          <th className="py-1">#</th>
          <th>Company</th>
          <th className="text-right">{label.split(":")[0]}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const c = view.companies.find((x) => x.id === r.companyId);
          if (!c) return null;
          const mine = r.companyId === me;
          return (
            <tr
              key={r.companyId}
              data-mine={mine}
              className={`border-t border-border ${mine ? "bg-brand-soft" : ""}`}
            >
              <td className="py-2 font-semibold">{r.rank}</td>
              <td>
                <button
                  type="button"
                  disabled={!c.rival}
                  onClick={(e) => open({ kind: "rival", companyId: c.id }, e.currentTarget)}
                  className="flex items-center gap-2 text-left disabled:cursor-default"
                >
                  <Logo logo={c.logo} palette={c.palette} size={24} />
                  <span style={mine ? { color: c.palette.primary } : undefined}>{c.name}</span>
                </button>
              </td>
              <td className="text-right font-mono">{r.text}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** Battles and league tables (game-layer design §3). */
export function Arena({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const [tab, setTab] = useState<Tab>("customers");
  const [skill, setSkill] = useState<BattleSkill>("knowledge");
  const battles = [...(view.me.company?.battles ?? [])].reverse();
  const league = view.world.league;
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
      <Panel title="League tables" hint="Every company, ranked by what was measured.">
        <div role="tablist" aria-label="Tables" className="mb-4 flex flex-wrap gap-1">
          {TABS.map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={`rounded-full px-3 py-1 text-xs ${tab === t ? "bg-brand text-on-brand" : "bg-surface-2 text-muted hover:text-text"}`}
            >
              {t === "points" ? "Ranking points" : TABLE_NAMES[t]}
            </button>
          ))}
        </div>
        {tab === "points" ? (
          <>
            <div className="mb-3 flex flex-wrap gap-1">
              {BATTLE_SKILLS.map((k) => (
                <button
                  key={k}
                  type="button"
                  aria-pressed={skill === k}
                  onClick={() => setSkill(k)}
                  className={`rounded-full px-2 py-0.5 text-xs ${skill === k ? "bg-surface-2 text-text" : "text-muted"}`}
                >
                  {k}
                </button>
              ))}
            </div>
            {league.points[skill].length === 0 ? (
              <p className="text-sm text-muted">No battles in {skill} yet.</p>
            ) : (
              <Rows
                view={view}
                label={`Ranking points: ${skill}`}
                rows={league.points[skill].map((r) => ({
                  companyId: r.companyId,
                  rank: r.rank,
                  text: String(r.points),
                }))}
              />
            )}
          </>
        ) : (
          <Rows
            view={view}
            label={TABLE_NAMES[tab]}
            rows={league.tables[tab].map((r) => ({
              companyId: r.companyId,
              rank: r.rank,
              text: valueText(tab, r.value),
            }))}
          />
        )}
      </Panel>
      <Panel
        title="Battles"
        hint="Your AI and a rival's answer the same fresh questions. Real answers decide."
        actions={
          <Button onClick={(e) => open({ kind: "newBattle" }, e.currentTarget)}>
            Start a battle
          </Button>
        }
      >
        {battles.length === 0 ? (
          <p className="text-sm text-muted">No battles yet.</p>
        ) : (
          <ul className="space-y-2 text-sm">
            {battles.map((b) => {
              const rival = view.companies.find((c) => c.id === b.rivalId);
              return (
                <li key={b.id}>
                  <button
                    type="button"
                    aria-label={`Open the ${b.skill} battle against ${rival?.name ?? "a rival"} on ${dayLabel(b.day)}`}
                    onClick={(e) => open({ kind: "battle", battleId: b.id }, e.currentTarget)}
                    className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-left hover:bg-surface-2"
                  >
                    <span>
                      {b.skill} vs {rival?.name ?? "a rival"}
                    </span>
                    <span className="text-muted">
                      {b.status === "running"
                        ? "on now"
                        : b.status === "calledOff"
                          ? "called off"
                          : b.outcome}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </Panel>
    </div>
  );
}
