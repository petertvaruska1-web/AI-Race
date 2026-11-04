import { type GoalSpec, NO_GOAL, TEXT_LIMITS } from "@airace/protocol";
import { type FormEvent, useEffect, useState } from "react";
import { Button } from "../components/Button.tsx";
import { Field, INPUT } from "../components/Field.tsx";
import { Panel } from "../components/Panel.tsx";
import { api, type Me, type WorldSummary } from "../net/api.ts";
import { go } from "../route.ts";
import { GoalPicker } from "../surprises/Goals.tsx";

/** First visit: a name is all a guest needs. */
export function Welcome({ onSignedIn }: { onSignedIn: (me: Me) => void }) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [min, max] = TEXT_LIMITS.displayName;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.guest(name);
      const me = await api.me();
      if (me) onSignedIn(me);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto mt-24 max-w-md px-6">
      <h1 className="mb-2 text-4xl font-bold">AI Race</h1>
      <p className="mb-8 text-muted">Run an AI company. Train a real AI. Race your friends.</p>
      <Panel>
        <form onSubmit={submit} className="space-y-4">
          <Field
            label="What should we call you?"
            hint={`${min}–${max} characters. Friends will see it.`}
          >
            <input
              className={INPUT}
              value={name}
              maxLength={max}
              onChange={(e) => setName(e.target.value)}
            />
          </Field>
          {error && (
            <p role="alert" className="text-sm text-danger">
              {error}
            </p>
          )}
          <Button type="submit" disabled={busy || name.trim().length < min}>
            Continue
          </Button>
        </form>
      </Panel>
    </main>
  );
}

export function Home({ me }: { me: Me }) {
  const [worlds, setWorlds] = useState<WorldSummary[] | null>(null);
  const [name, setName] = useState("");
  const [goal, setGoal] = useState<GoalSpec>(NO_GOAL);
  const [rivals, setRivals] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [min, max] = TEXT_LIMITS.worldName;

  useEffect(() => {
    api
      .listWorlds()
      .then(setWorlds)
      .catch((err: Error) => setError(err.message));
    api
      .setup()
      .then((s) => setRivals(s.rivals))
      .catch(() => setRivals(false));
  }, []);

  async function create(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      const worldId = await api.createWorld(name, goal.goal.kind === "none" ? undefined : goal);
      go({ screen: "world", worldId, tab: "company" });
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <main className="mx-auto max-w-3xl space-y-6 px-6 py-10">
      <header>
        <h1 className="text-3xl font-bold">Welcome, {me.displayName}</h1>
        <p className="text-muted">Pick up a world or start a new one.</p>
      </header>
      <Panel title="Your worlds">
        {worlds === null ? (
          <p className="text-muted">Loading…</p>
        ) : worlds.length === 0 ? (
          <p className="text-muted">No worlds yet. Start one below.</p>
        ) : (
          <ul className="divide-y divide-border">
            {worlds.map((w) =>
              w.outdated ? (
                <li key={w.id} className="py-3 opacity-70">
                  <p className="player-text font-semibold">{w.name}</p>
                  <p className="text-sm text-muted">Made with an older version of the game</p>
                </li>
              ) : (
                <li key={w.id} className="flex items-center justify-between py-3">
                  <div>
                    <p className="player-text font-semibold">{w.name}</p>
                    <p className="text-sm text-muted">
                      {w.phase === "setup" ? "Not started yet" : `Day ${w.day}`} · {w.memberCount}{" "}
                      {w.memberCount === 1 ? "player" : "players"}
                    </p>
                  </div>
                  <Button
                    variant="secondary"
                    onClick={() => go({ screen: "world", worldId: w.id, tab: "company" })}
                  >
                    Open
                  </Button>
                </li>
              ),
            )}
          </ul>
        )}
      </Panel>
      <Panel
        title="Start a new world"
        hint="A world is one game. Its clock starts when you open for business."
      >
        <form onSubmit={create} className="space-y-4">
          <div className="flex gap-3">
            <input
              className={INPUT}
              aria-label="World name"
              placeholder="Friday night"
              value={name}
              maxLength={max}
              onChange={(e) => setName(e.target.value)}
            />
            <Button type="submit" disabled={name.trim().length < min}>
              Start
            </Button>
          </div>
          <GoalPicker rivals={rivals} onChange={setGoal} />
        </form>
      </Panel>
      {error && (
        <p role="alert" className="text-sm text-danger">
          {error}
        </p>
      )}
    </main>
  );
}
