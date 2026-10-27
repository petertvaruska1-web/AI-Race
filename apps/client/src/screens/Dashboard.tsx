import { SPEEDS, type Speed, type WorldView } from "@airace/protocol";
import type { ReactNode } from "react";
import { Button } from "../components/Button.tsx";
import { Card } from "../components/Card.tsx";
import { Logo } from "../components/Logo.tsx";
import { formatMoney } from "../format.ts";
import { go, type Route } from "../route.ts";
import { useSocket } from "../store/session.ts";

type Tab = Extract<Route, { screen: "world" }>["tab"];

const TABS: { id: Tab; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "lab", label: "AI Lab" },
  { id: "play", label: "Playground" },
  { id: "bench", label: "Benchmarks" },
];

/** The world's frame: the company in its own colors, its cash, the clock and the navigation. */
export function WorldLayout({
  view,
  tab,
  children,
}: {
  view: WorldView;
  tab: Tab;
  children: ReactNode;
}) {
  const company = view.companies.find((c) => c.id === view.me.company?.id);
  return (
    <div className="min-h-screen">
      <header
        className="border-b border-border bg-surface"
        style={{ borderTop: "4px solid var(--brand)" }}
      >
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-4 px-6 py-3">
          <a href="#/" className="text-sm text-muted hover:text-text" aria-label="All worlds">
            ←
          </a>
          {company && (
            <Logo logo={company.logo} palette={company.palette} size={36} title={company.name} />
          )}
          <div className="mr-auto">
            <p className="player-text font-display text-lg font-semibold text-brand">
              {company?.name}
            </p>
            <p className="player-text text-xs text-muted">{view.world.name}</p>
          </div>
          <Clock view={view} />
          <p className="text-right">
            <span className="block text-xs text-muted">Cash</span>
            <span className="font-semibold">{formatMoney(view.me.company?.cash ?? 0)}</span>
          </p>
        </div>
        <nav className="mx-auto flex max-w-6xl gap-1 px-6" aria-label="Sections">
          {TABS.map((t) => (
            <a
              key={t.id}
              href={`#/w/${view.world.id}/${t.id}`}
              aria-current={tab === t.id ? "page" : undefined}
              className={`border-b-2 px-3 py-2 text-sm ${tab === t.id ? "border-brand text-text" : "border-transparent text-muted hover:text-text"}`}
            >
              {t.label}
            </a>
          ))}
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
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
      <p className="text-right">
        <span className="block text-xs text-muted">Day {view.world.day}</span>
        <span className="font-semibold">{view.world.date}</span>
      </p>
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

const RUN_STATUS: Record<string, string> = {
  training: "Training",
  finalizing: "Finishing up",
  completed: "Done",
  failed: "Failed",
};

export function Dashboard({ view }: { view: WorldView }) {
  const company = view.me.company;
  if (!company) return null;
  const run = company.runs.at(-1);
  const latest = company.versions.at(-1);
  const lab = () => go({ screen: "world", worldId: view.world.id, tab: "lab" });
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card title="Training">
        {run ? (
          <div className="space-y-2">
            <p>
              <span className="font-semibold">{RUN_STATUS[run.status]}</span>
              {run.status === "training" && !run.computeStarted && (
                <span className="text-muted"> · waiting for a lab computer</span>
              )}
            </p>
            {(run.status === "training" || run.status === "finalizing") && (
              <progress
                className="h-2 w-full accent-[var(--brand)]"
                aria-label="Training progress"
                value={run.progress}
                max={1}
              />
            )}
            <Button variant="secondary" onClick={lab}>
              Open the AI Lab
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-muted">
              You have no AI yet. Plan and train your first one in the AI Lab.
            </p>
            <Button onClick={lab}>Train your first AI</Button>
          </div>
        )}
      </Card>
      <Card title="Your latest AI">
        {latest ? (
          <div className="space-y-1">
            <p className="text-2xl font-bold text-brand">{latest.label}</p>
            <p className="text-sm text-muted">
              {latest.overall === null
                ? "Not scored yet"
                : `Overall score ${Math.round(latest.overall)}`}
            </p>
            {latest.personalityWords.length > 0 && (
              <p className="text-sm">{latest.personalityWords.join(", ")}</p>
            )}
          </div>
        ) : (
          <p className="text-muted">Nothing trained yet.</p>
        )}
      </Card>
    </div>
  );
}
