import { SPEEDS, type Speed, type WorldView } from "@airace/protocol";
import { getBalance } from "@airace/sim";
import type { ReactNode } from "react";
import { formatMoney } from "../format.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { CountUp } from "./CountUp.tsx";
import { FloatDelta } from "./FloatDelta.tsx";
import { Logo } from "./Logo.tsx";

const total = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);

/** This month's income less its costs. */
export function monthProfit(view: WorldView): number {
  const month = view.me.company?.ledger.current;
  return month ? total(month.income) - total(month.costs) : 0;
}

function Figure({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="block text-right">
      <span className="block text-[11px] font-semibold uppercase tracking-wider text-muted">
        {label}
      </span>
      <span className="block font-display text-lg font-semibold">{children}</span>
    </span>
  );
}

/** A thin line that fills over each game day at the world's speed; it rests while paused. */
function DayBar({ view }: { view: WorldView }) {
  const ms = getBalance(view.world.balance).tickMsAt1x / view.world.speed;
  return (
    <div aria-hidden="true" className="h-0.5 w-full overflow-hidden">
      <div
        key={view.world.day}
        data-testid="day-bar"
        data-paused={view.world.paused}
        className="day-bar h-full w-full"
        style={{ animationDuration: `${ms}ms` }}
      />
    </div>
  );
}

/** The company in its colors, its money at a glance, and the shared clock. */
export function TopBar({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const company = view.companies.find((c) => c.id === view.me.company?.id);
  const profit = monthProfit(view);
  const cash = view.me.company?.cash ?? 0;
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
        <button
          type="button"
          aria-label="Open money"
          onClick={(e) => open({ kind: "money" }, e.currentTarget)}
          className="lift flex gap-6 rounded-lg px-2 py-1"
        >
          <Figure label="Cash">
            <span className="relative">
              <CountUp value={cash} format={formatMoney} />
              <FloatDelta value={cash} format={formatMoney} />
            </span>
          </Figure>
          <Figure label="Profit this month">
            <span className={profit < 0 ? "text-danger" : profit > 0 ? "text-good" : ""}>
              <CountUp value={profit} format={formatMoney} />
            </span>
          </Figure>
        </button>
        <Clock view={view} />
      </div>
      <DayBar view={view} />
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
