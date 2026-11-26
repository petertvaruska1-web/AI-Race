import {
  type GoalSpec,
  INVITE_ALPHABET,
  INVITE_LENGTH,
  NO_GOAL,
  TEXT_LIMITS,
} from "@airace/protocol";
import { MotionGlobalConfig, motion } from "motion/react";
import { type FormEvent, useEffect, useState } from "react";
import { Display } from "../bench/Display.tsx";
import { Keypad } from "../bench/Keypad.tsx";
import { Lamp, type LampState } from "../bench/Lamp.tsx";
import { Module } from "../bench/Module.tsx";
import { PushButton } from "../bench/PushButton.tsx";
import { ToggleSwitch } from "../bench/ToggleSwitch.tsx";
import { FAIR_NOTICE } from "../fair.ts";
import { api, type Me, type WorldSummary } from "../net/api.ts";
import { go } from "../route.ts";
import { GoalPicker } from "../surprises/Goals.tsx";

/** How long the ID badge takes to print and slide out of the slot (C62). */
const PRINT_MS = 700;

const wait = (ms: number) => new Promise<void>((done) => setTimeout(done, ms));

/**
 * The lab door's ID badge printer (C62): a guest types a name into its display and "Continue"
 * prints a badge with that name, which slides out of the slot while the player is signed in.
 */
export function Welcome({
  onSignedIn,
  intro = "Run an AI company. Train a real AI. Race your friends.",
}: {
  onSignedIn: (me: Me) => void;
  intro?: string;
}) {
  const [name, setName] = useState("");
  const [printed, setPrinted] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [min, max] = TEXT_LIMITS.displayName;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setPrinted(name.trim());
    try {
      // The badge prints while the sign-in runs; no wait at all for less motion.
      const [me] = await Promise.all([
        api.guest(name).then(() => api.me()),
        wait(MotionGlobalConfig.skipAnimations ? 0 : PRINT_MS),
      ]);
      if (me) onSignedIn(me);
    } catch (err) {
      setError((err as Error).message);
      setPrinted(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="door-frame door-frame--welcome">
      <div className="welcome">
        <header className="welcome__head">
          <h1 className="welcome__title">AI Race</h1>
          <p className="welcome__intro">{intro}</p>
        </header>
        <Module weight="heavy" className="printer">
          <form onSubmit={submit} className="printer__form">
            <label className="printer__field">
              <span className="engrave">What should we call you?</span>
              <input
                className="display-input printer__display"
                value={name}
                maxLength={max}
                autoComplete="off"
                spellCheck={false}
                onChange={(e) => setName(e.target.value)}
              />
            </label>
            <p className="module__hint">{`${min}–${max} characters. Friends will see it.`}</p>
            {error && (
              <p role="alert" className="text-sm text-danger">
                {error}
              </p>
            )}
            <PushButton type="submit" disabled={busy || name.trim().length < min}>
              Continue
            </PushButton>
          </form>
          <div className="printer__slot">
            {printed !== null && (
              <motion.div
                data-testid="id-badge"
                className="id-badge"
                initial={{ y: "-104%" }}
                animate={{ y: 0 }}
                transition={{ duration: PRINT_MS / 1000, ease: [0.22, 0.9, 0.3, 1] }}
              >
                <span className="id-badge__band">
                  <span className="engrave">AI Race</span>
                  <span className="engrave">Lab access</span>
                </span>
                <span className="dotface id-badge__name">{printed}</span>
                <span className="id-badge__code" aria-hidden="true" />
              </motion.div>
            )}
          </div>
        </Module>
      </div>
    </main>
  );
}

type PanelState = "running" | "setup" | "outdated";

function stateOf(w: WorldSummary): PanelState {
  return w.outdated ? "outdated" : w.phase === "setup" ? "setup" : "running";
}

const LAMP: Record<PanelState, { state: LampState; label: string }> = {
  running: { state: "on", label: "Running" },
  setup: { state: "wait", label: "Not started" },
  outdated: { state: "off", label: "Older version" },
};

/** One world on the rack: lit while it runs, amber before it starts, dim from an older version. */
function WorldPanel({ world }: { world: WorldSummary }) {
  const state = stateOf(world);
  return (
    <li data-state={state} className={`plate-screws rack-panel rack-panel--${state}`}>
      <div className="rack-panel__head">
        <Lamp state={LAMP[state].state} label={LAMP[state].label} />
        <span className="player-text rack-panel__name" title={world.name}>
          {world.name}
        </span>
      </div>
      {world.outdated ? (
        <p className="rack-panel__note">Made with an older version of the game</p>
      ) : (
        <>
          <Display className="rack-panel__day">
            <span className="dotface dotface--sm">
              {world.phase === "setup" ? "Not started yet" : `Day ${world.day}`}
            </span>
          </Display>
          <div className="rack-panel__foot">
            <span className="rack-panel__players">
              {world.memberCount} {world.memberCount === 1 ? "player" : "players"}
            </span>
            <PushButton
              variant="secondary"
              onClick={() => go({ screen: "world", worldId: world.id, tab: "company" })}
            >
              Open
            </PushButton>
          </div>
        </>
      )}
    </li>
  );
}

/** Leaves for a friend's invite link (a fresh page at its address). */
function openJoinLink(path: string): void {
  location.assign(path);
}

/**
 * Behind the lab door (C62): the player's worlds hang on a wall rack, a new world is set up on a
 * small bench, and a friend's world is joined with the keypad.
 */
export function Home({
  me,
  openLink = openJoinLink,
}: {
  me: Me;
  openLink?: (path: string) => void;
}) {
  const [worlds, setWorlds] = useState<WorldSummary[] | null>(null);
  const [name, setName] = useState("");
  const [goal, setGoal] = useState<GoalSpec>(NO_GOAL);
  const [rivals, setRivals] = useState(false);
  const [fairMode, setFairMode] = useState(false);
  const [code, setCode] = useState("");
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
      const worldId = await api.createWorld(
        name,
        goal.goal.kind === "none" ? undefined : goal,
        fairMode,
      );
      go({ screen: "world", worldId, tab: "company" });
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <main className="door-frame">
      <header className="door-strip module module--thin">
        <div className="min-w-0">
          <h1 className="m-0 font-display text-[25px] font-bold">
            Welcome, <span className="player-text">{me.displayName}</span>
          </h1>
          <p className="m-0 text-sm text-muted">Pick up a world or start a new one.</p>
        </div>
        {error && (
          <p role="alert" className="ml-auto text-sm text-danger">
            {error}
          </p>
        )}
      </header>

      <Module label="Your worlds" className="door-rack">
        {worlds === null ? (
          <p className="text-muted">Loading…</p>
        ) : worlds.length === 0 ? (
          <p className="text-muted">No worlds yet. Start one on the bench.</p>
        ) : (
          <ul aria-label="Your worlds" className="rack">
            {worlds.map((w) => (
              <WorldPanel key={w.id} world={w} />
            ))}
          </ul>
        )}
      </Module>

      <div className="door-side">
        <Module
          label="Start a new world"
          hint="A world is one game. Its clock starts when you open for business."
        >
          <form onSubmit={create} className="new-world">
            <input
              className="display-input door-plate"
              aria-label="World name"
              placeholder="Friday night"
              value={name}
              maxLength={max}
              autoComplete="off"
              spellCheck={false}
              onChange={(e) => setName(e.target.value)}
            />
            <GoalPicker rivals={rivals} onChange={setGoal} />
            <div className="new-world__fair">
              <ToggleSwitch label="Fair mode" checked={fairMode} onChange={setFairMode} />
              {fairMode ? (
                <p className="notice-plate">{FAIR_NOTICE}</p>
              ) : (
                <p className="m-0 text-sm text-muted">
                  Each player's AI trains on their own computer. A faster graphics card is an edge.
                </p>
              )}
            </div>
            <PushButton type="submit" disabled={name.trim().length < min}>
              Start
            </PushButton>
          </form>
        </Module>
        <Module label="Join a friend" hint="Your friend's world has an invite code in its lobby.">
          <form
            className="join-pad"
            onSubmit={(e) => {
              e.preventDefault();
              openLink(`/join/${encodeURIComponent(code)}`);
            }}
          >
            <Keypad
              label="Invite code"
              alphabet={INVITE_ALPHABET}
              value={code}
              maxLength={INVITE_LENGTH}
              onChange={setCode}
            />
            <PushButton type="submit" variant="secondary" disabled={code.length === 0}>
              Join
            </PushButton>
          </form>
        </Module>
      </div>
    </main>
  );
}
