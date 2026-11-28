import {
  BATTLE_SKILLS,
  type BattleSkill,
  BENCH_CATEGORIES,
  type LeagueTable,
  type WorldView,
} from "@airace/protocol";
import { useState } from "react";
import { Display } from "../bench/Display.tsx";
import { KeyRow } from "../bench/KeyRow.tsx";
import { Module } from "../bench/Module.tsx";
import { PushButton } from "../bench/PushButton.tsx";
import { RotarySelector } from "../bench/RotarySelector.tsx";
import { Logo } from "../components/Logo.tsx";
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

/** One league table on the scoreboard: rank and value in the dot face, the company between. */
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
    <table aria-label={label} className="scoreboard">
      <thead>
        <tr>
          <th className="scoreboard__rank">#</th>
          <th>Company</th>
          <th className="scoreboard__value">{label.split(":")[0]}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const c = view.companies.find((x) => x.id === r.companyId);
          if (!c) return null;
          const mine = r.companyId === me;
          return (
            <tr key={r.companyId} data-mine={mine} className="scoreboard__row">
              <td className="scoreboard__rank">
                <span className={`dotface dotface--sm ${r.rank === 1 ? "" : "dotface--plain"}`}>
                  {r.rank}
                </span>
              </td>
              <td>
                <button
                  type="button"
                  disabled={!c.rival}
                  onClick={(e) => open({ kind: "rival", companyId: c.id }, e.currentTarget)}
                  className="scoreboard__company"
                >
                  <Logo logo={c.logo} palette={c.palette} size={24} />
                  <span>{c.name}</span>
                  {mine && <span className="scoreboard__you">you</span>}
                </button>
              </td>
              <td className="scoreboard__value">
                <span className="dotface dotface--xs">{r.text}</span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** Battles and league tables (game-layer design §3), on the Arena bench. */
export function Arena({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const [tab, setTab] = useState<Tab>("customers");
  const [skill, setSkill] = useState<BattleSkill>("knowledge");
  const battles = [...(view.me.company?.battles ?? [])].reverse();
  const league = view.world.league;
  return (
    <div className="arena-bench">
      <Module
        label="League tables"
        hint="Every company, ranked by what was measured."
        weight="heavy"
        className="arena-league"
      >
        <div className="arena-league__body">
          <div className="arena-league__dials">
            <RotarySelector
              legend="Tables"
              options={TABS.map((t) => ({
                value: t,
                label: t === "points" ? "Ranking points" : TABLE_NAMES[t],
              }))}
              value={tab}
              onChange={setTab}
              size={58}
            />
            {tab === "points" && (
              <KeyRow
                legend="Battle skill"
                className="arena-skills"
                options={BATTLE_SKILLS.map((k) => ({ value: k, label: TABLE_NAMES[k] }))}
                value={skill}
                onChange={setSkill}
              />
            )}
          </div>
          <div className="display display--scroll arena-board">
            {tab === "points" ? (
              league.points[skill].length === 0 ? (
                <p className="arena-board__empty">
                  No battles in {TABLE_NAMES[skill].toLowerCase()} yet.
                </p>
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
              )
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
          </div>
        </div>
      </Module>
      <Module
        label="Battles"
        hint="Your AI and a rival's answer the same fresh questions. Real answers decide."
        className="arena-battles"
      >
        <PushButton
          className="arena-battles__start"
          onClick={(e) => open({ kind: "newBattle" }, e.currentTarget)}
        >
          Start a battle
        </PushButton>
        {battles.length === 0 ? (
          <p className="arena-board__empty">No battles yet.</p>
        ) : (
          <Display scroll label="Your battles" className="arena-battles__list">
            <ul>
              {battles.map((b) => {
                const rival = view.companies.find((c) => c.id === b.rivalId);
                return (
                  <li key={b.id}>
                    <button
                      type="button"
                      aria-label={`Open the ${b.skill} battle against ${rival?.name ?? "a rival"} on ${dayLabel(b.day)}`}
                      onClick={(e) => open({ kind: "battle", battleId: b.id }, e.currentTarget)}
                      className="arena-battle"
                    >
                      <span>
                        {b.defending
                          ? `${b.skill}: ${rival?.name ?? "a player"} challenged you`
                          : `${b.skill} vs ${rival?.name ?? "a rival"}`}
                      </span>
                      <span className="arena-battle__state">
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
          </Display>
        )}
      </Module>
    </div>
  );
}
