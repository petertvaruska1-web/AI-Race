import { SPEEDS, type Speed, type WorldView } from "@airace/protocol";
import { getBalance } from "@airace/sim";
import { Readout } from "../bench/Readout.tsx";
import { ToggleSwitch } from "../bench/ToggleSwitch.tsx";
import { CardsChip } from "../cards/Cards.tsx";
import { formatMoney } from "../format.ts";
import { LogoMotion } from "../spending/Spending.tsx";
import { useGraphics } from "../store/graphics.ts";
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

/** A thin line along the strip's lower edge that fills over each game day; it rests while paused. */
function DayBar({ view }: { view: WorldView }) {
  const ms = getBalance(view.world.balance).tickMsAt1x / view.world.speed;
  return (
    <div aria-hidden="true" className="station-strip__day">
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

/** "Lighter graphics" (design §11), and why it came on when the game switched it on itself. */
export function LighterGraphics() {
  const lighter = useGraphics((s) => s.lighter);
  const auto = useGraphics((s) => s.auto);
  const setLighter = useGraphics((s) => s.setLighter);
  return (
    <div className="flex flex-col items-start gap-0.5">
      <ToggleSwitch label="Lighter graphics" checked={lighter} onChange={(v) => setLighter(v)} />
      {auto && (
        <span aria-live="polite" className="text-[11.5px] text-muted">
          Switched on to keep the game smooth
        </span>
      )}
    </div>
  );
}

/** The top strip (design §7.1): the company, its money at a glance, the shared clock. */
export function TopBar({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const company = view.companies.find((c) => c.id === view.me.company?.id);
  const profit = monthProfit(view);
  const cash = view.me.company?.cash ?? 0;
  const served = view.me.company?.compute.served ?? 1;
  return (
    <header className="station-strip module module--thin">
      <a href="#/" className="icon-btn" aria-label="All worlds">
        ←
      </a>
      {company && (
        <LogoMotion motion={company.logoMotion}>
          <Logo logo={company.logo} palette={company.palette} size={40} />
        </LogoMotion>
      )}
      <div className="station-strip__name">
        <p
          className="truncate font-display text-[21px] font-bold leading-tight text-brand"
          title={company?.name}
        >
          {company?.name}
        </p>
        <p className="truncate text-[13px] text-muted" title={view.world.name}>
          {view.world.name}
        </p>
      </div>
      <button
        type="button"
        aria-label="Open money"
        onClick={(e) => open({ kind: "money" }, e.currentTarget)}
        className="lift station-strip__money"
      >
        <Readout
          label="Cash"
          value={
            <span className="relative">
              <CountUp value={cash} format={formatMoney} />
              <FloatDelta value={cash} format={formatMoney} />
            </span>
          }
        />
        <Readout
          label="Profit this month"
          tone={profit < 0 ? "danger" : profit > 0 ? "good" : "lamp"}
          value={<CountUp value={profit} format={formatMoney} />}
        />
      </button>
      {served < 1 && (
        <button
          type="button"
          aria-label={`Servers overloaded · ${Math.round(served * 100)}% answered`}
          onClick={(e) => open({ kind: "computers" }, e.currentTarget)}
          className="overload-chip push push--secondary text-danger"
        >
          {`Servers overloaded · ${Math.round(served * 100)}% answered`}
        </button>
      )}
      <CardsChip view={view} />
      <Clock view={view} />
      <DayBar view={view} />
    </header>
  );
}

/** The date, and for the host the speed and pause keys (C1: one shared clock). */
function Clock({ view }: { view: WorldView }) {
  const socket = useSocket();
  const isHost = view.world.effectiveHostUserId === view.me.userId;
  const send = (cmd: { type: "setSpeed"; speed: Speed } | { type: "setPaused"; paused: boolean }) =>
    void socket.command(view.world.id, cmd);
  return (
    <div className="flex items-end gap-2.5">
      <Readout label={`Day ${view.world.day}`} value={view.world.date} />
      {isHost && (
        <fieldset className="m-0 flex gap-1 border-0 p-0 pb-0.5" aria-label="Game speed">
          <button
            type="button"
            onClick={() => send({ type: "setPaused", paused: !view.world.paused })}
            className={`speed-key ${view.world.paused ? "speed-key--on" : ""}`}
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
              className={`speed-key ${view.world.speed === s && !view.world.paused ? "speed-key--on" : ""}`}
            >
              {s}×
            </button>
          ))}
        </fieldset>
      )}
    </div>
  );
}
