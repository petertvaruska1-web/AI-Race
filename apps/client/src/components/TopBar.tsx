import { SPEEDS, type Speed, type WorldView } from "@airace/protocol";
import type { ReactNode } from "react";
import { formatMoney } from "../format.ts";
import { useSocket } from "../store/session.ts";
import { CountUp } from "./CountUp.tsx";
import { Logo } from "./Logo.tsx";

const total = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);

/** This month's income less its costs. */
export function monthProfit(view: WorldView): number {
  const month = view.me.company?.ledger.current;
  return month ? total(month.income) - total(month.costs) : 0;
}

function Figure({ label, children }: { label: string; children: ReactNode }) {
  return (
    <p className="text-right">
      <span className="block text-[11px] font-semibold uppercase tracking-wider text-muted">
        {label}
      </span>
      <span className="font-display text-lg font-semibold">{children}</span>
    </p>
  );
}

/** The company in its colors, its money at a glance, and the shared clock. */
export function TopBar({ view }: { view: WorldView }) {
  const company = view.companies.find((c) => c.id === view.me.company?.id);
  const profit = monthProfit(view);
  return (
    <header className="sticky top-0 z-20 border-b border-border bg-bg/80 backdrop-blur-md">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 px-5 py-3">
        <a href="#/" className="text-sm text-muted hover:text-text" aria-label="All worlds">
          ←
        </a>
        {company && <Logo logo={company.logo} palette={company.palette} size={36} />}
        <div className="mr-auto min-w-0">
          <p className="player-text font-display text-lg font-semibold text-brand">
            {company?.name}
          </p>
          <p className="player-text text-xs text-muted">{view.world.name}</p>
        </div>
        <Figure label="Cash">
          <CountUp value={view.me.company?.cash ?? 0} format={formatMoney} />
        </Figure>
        <Figure label="Profit this month">
          <span className={profit < 0 ? "text-danger" : profit > 0 ? "text-good" : ""}>
            <CountUp value={profit} format={formatMoney} />
          </span>
        </Figure>
        <Clock view={view} />
      </div>
    </header>
  );
}

/** The date, and for the host the speed and pause controls (C1: one shared clock). */
function Clock({ view }: { view: WorldView }) {
  const socket = useSocket();
  const isHost = view.world.effectiveHostUserId === view.me.userId;
  const send = (cmd: { type: "setSpeed"; speed: Speed } | { type: "setPaused"; paused: boolean }) =>
    void socket.command(view.world.id, cmd);
  return (
    <div className="flex items-center gap-2">
      <Figure label={`Day ${view.world.day}`}>{view.world.date}</Figure>
      {isHost && (
        <fieldset className="m-0 flex gap-1 border-0 p-0" aria-label="Game speed">
          <button
            type="button"
            onClick={() => send({ type: "setPaused", paused: !view.world.paused })}
            className="rounded-md bg-surface-2 px-2 py-1 text-xs"
            aria-label={view.world.paused ? "Resume" : "Pause"}
          >
            {view.world.paused ? "▶" : "❚❚"}
          </button>
          {SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              aria-pressed={view.world.speed === s}
              onClick={() => send({ type: "setSpeed", speed: s })}
              className={`rounded-md px-2 py-1 text-xs ${view.world.speed === s ? "bg-brand text-on-brand" : "bg-surface-2 text-muted"}`}
            >
              {s}×
            </button>
          ))}
        </fieldset>
      )}
    </div>
  );
}
