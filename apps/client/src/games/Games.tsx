import {
  DIFFICULTIES,
  type Difficulty,
  DUEL_GAMES,
  GAME_NAMES,
  GAME_STEPS,
  GAMES,
  type GameId,
  type MinigameSettings,
  QUIZ_TOPICS,
  type QuizTopic,
  STORY_HEROES,
  STORY_SITUATIONS,
  STORY_WORDS,
  type WorldView,
} from "@airace/protocol";
import { formatMoney } from "@airace/sim";
import { motion } from "motion/react";
import { useState } from "react";
import { Button } from "../components/Button.tsx";
import { INPUT } from "../components/Field.tsx";
import { Panel } from "../components/Panel.tsx";
import { Toggle } from "../components/Toggle.tsx";
import { capitalize, dayLabel } from "../format.ts";
import { rivalAis } from "../rivals.ts";
import { useSocket } from "../store/session.ts";
import { DICE } from "./pictures.tsx";
import { depthOf, Lives, STAGES } from "./Stages.tsx";
import { type LiveRun, useMinigame } from "./useMinigame.ts";

export const GAME_INFO: Record<GameId, { icon: string; skill: string; blurb: string }> = {
  maze: {
    icon: "🧩",
    skill: "Reasoning",
    blurb: "Your AI picks a door at every junction by solving a puzzle. Right goes deeper.",
  },
  bridge: {
    icon: "🌉",
    skill: "Patterns",
    blurb: "Each tile is a 'what comes next'. A right answer holds; a wrong one crumbles.",
  },
  quiz: {
    icon: "🎤",
    skill: "Knowledge",
    blurb: "A game show: four answers, a prize ladder, and how sure your AI is of each.",
  },
  story: {
    icon: "🎲",
    skill: "Creativity and rules",
    blurb: "Roll a hero, a situation and a word it must use. The audience judges the story.",
  },
  simon: {
    icon: "🗣️",
    skill: "Following instructions",
    blurb: "Commands flip up one by one. Green for done right, red for not quite.",
  },
};

const DIFFICULTY_WORDS: Record<Difficulty, string> = {
  easy: "Easy: 2 choices",
  medium: "Medium: 3 choices",
  hard: "Hard: 4 choices, and it types late answers",
};

type Choice = { game: GameId; difficulty: Difficulty };

/** The Playground's games: pick one, set it up, watch your AI play it (playtest pack §4). */
export function Games({ view, versionId }: { view: WorldView; versionId: string }) {
  const { run, start, follow, reset, history, loadHistory } = useMinigame(view.world.id);
  const [choice, setChoice] = useState<Choice | null>(null);
  const [opponent, setOpponent] = useState<string | null>(null);
  if (run.status !== "idle" && choice) {
    return (
      <PlayView
        view={view}
        game={choice.game}
        difficulty={choice.difficulty}
        run={run}
        opponent={opponent}
        best={history.bests[`${choice.game}:${choice.difficulty}`] ?? null}
        onBack={() => {
          reset();
          setChoice(null);
        }}
      />
    );
  }
  if (choice) {
    return (
      <GameSetup
        view={view}
        game={choice.game}
        versionId={versionId}
        onBack={() => setChoice(null)}
        onDifficulty={(difficulty) => setChoice({ ...choice, difficulty })}
        onPractice={(o) => {
          setOpponent(o.opponent);
          void start(o);
        }}
        onShowcase={async () => {
          // The world plays a showcase; its run appears in the player's runs to follow.
          for (let k = 0; k < 20; k++) {
            const h = await loadHistory();
            const live = h?.runs.find((r) => r.showcase && r.status === "running");
            if (live) return follow(live.id);
            await new Promise((r) => setTimeout(r, 1000));
          }
        }}
      />
    );
  }
  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {GAMES.map((g) => (
        <motion.button
          key={g}
          type="button"
          whileHover={{ y: -3 }}
          onClick={() => setChoice({ game: g, difficulty: "easy" })}
          className="panel p-4 text-left"
          aria-label={`Play ${GAME_NAMES[g]}`}
        >
          <span className="text-4xl">{GAME_INFO[g].icon}</span>
          <h3 className="mt-2 font-display text-lg font-semibold">{GAME_NAMES[g]}</h3>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted">
            {GAME_INFO[g].skill}
          </p>
          <p className="mt-2 text-sm text-muted">{GAME_INFO[g].blurb}</p>
          <p className="mt-3 text-xs">
            {DIFFICULTIES.map((d) => {
              const best = history.bests[`${g}:${d}`];
              return best === undefined ? null : (
                <span key={d} className="mr-3" data-testid={`best-${g}-${d}`}>
                  {`${capitalize(d)} best: ${Math.round(best)}`}
                </span>
              );
            })}
          </p>
        </motion.button>
      ))}
    </div>
  );
}

/** Story Dice settings: three rolls, each a hero, a situation and a word (or none). */
function DiceSetup({
  rolls,
  onChange,
}: {
  rolls: [number, number, number | null][];
  onChange: (r: [number, number, number | null][]) => void;
}) {
  const roll = () =>
    onChange(
      rolls.map(() => [
        Math.floor(Math.random() * STORY_HEROES.length),
        Math.floor(Math.random() * STORY_SITUATIONS.length),
        Math.random() < 0.8 ? Math.floor(Math.random() * STORY_WORDS.length) : null,
      ]),
    );
  return (
    <div className="space-y-2">
      <Button variant="secondary" onClick={roll}>
        🎲 Roll the dice
      </Button>
      {rolls.map((r, k) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: three fixed story slots
        <div key={k} className="flex flex-wrap items-center gap-2 text-sm">
          <span className="w-16 text-muted">{`Story ${k + 1}`}</span>
          {(["hero", "situation", "word"] as const).map((die, i) => (
            <select
              key={die}
              aria-label={`Story ${k + 1} ${die}`}
              className={`${INPUT} w-auto`}
              value={r[i] ?? ""}
              onChange={(e) => {
                const next = rolls.map((x) => [...x]) as [number, number, number | null][];
                const v = e.target.value === "" ? null : Number(e.target.value);
                (next[k] as (number | null)[])[i] = v;
                onChange(next);
              }}
            >
              {die === "word" && <option value="">no rule</option>}
              {DICE[die].map((f, j) => (
                <option key={f.label} value={j}>{`${f.emoji} ${f.label}`}</option>
              ))}
            </select>
          ))}
        </div>
      ))}
    </div>
  );
}

function GameSetup({
  view,
  game,
  versionId,
  onBack,
  onDifficulty,
  onPractice,
  onShowcase,
}: {
  view: WorldView;
  game: GameId;
  versionId: string;
  onBack: () => void;
  onDifficulty: (d: Difficulty) => void;
  onPractice: (o: {
    game: GameId;
    difficulty: Difficulty;
    settings: MinigameSettings;
    versionId: string;
    opponent: string | null;
  }) => void;
  onShowcase: () => Promise<void>;
}) {
  const socket = useSocket();
  const company = view.me.company;
  const [difficulty, setDifficulty] = useState<Difficulty>("easy");
  const [topic, setTopic] = useState<QuizTopic>("mixed");
  const [rolls, setRolls] = useState<[number, number, number | null][]>([
    [0, 0, 0],
    [3, 2, 4],
    [8, 10, null],
  ]);
  const [opponent, setOpponent] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const rivals = rivalAis(view);
  const duel = (DUEL_GAMES as readonly string[]).includes(game);
  const settings: MinigameSettings =
    game === "quiz" ? { topic } : game === "story" ? { rolls } : {};
  const live = company?.products.find((p) => p.status === "live" && p.versionId === versionId);
  const shop = company?.shop.showcase;
  const lastShow = company?.showcases.at(-1);
  const nextDay = lastShow && shop ? lastShow.day + shop.cooldownDays : 0;
  const showcaseRunning = company?.showcases.some((x) => x.status === "running") ?? false;
  const showcaseReason = !live
    ? "Put this AI on sale to show it off."
    : showcaseRunning
      ? "Your last showcase is still going."
      : view.world.day < nextDay
        ? `The press saw you recently. Next on ${dayLabel(nextDay)}.`
        : null;
  return (
    <Panel
      title={`${GAME_INFO[game].icon} ${GAME_NAMES[game]}`}
      hint={GAME_INFO[game].blurb}
      actions={
        <Button variant="secondary" onClick={onBack}>
          All games
        </Button>
      }
    >
      <div className="space-y-4">
        {game !== "story" && (
          <div>
            <Toggle
              legend="Difficulty"
              options={DIFFICULTIES.map((d) => ({ label: capitalize(d), value: d }))}
              value={difficulty}
              onChange={(d) => {
                setDifficulty(d);
                onDifficulty(d);
              }}
            />
            <p className="text-xs text-muted">
              {game === "simon"
                ? "Harder starts with harder commands. It gets harder as it goes either way."
                : `${DIFFICULTY_WORDS[difficulty]}. It gets harder as it goes.`}
            </p>
          </div>
        )}
        {game === "quiz" && (
          <label className="block text-sm">
            <span className="mb-1 block font-semibold">Topic</span>
            <select
              className={INPUT}
              value={topic}
              onChange={(e) => setTopic(e.target.value as QuizTopic)}
            >
              {QUIZ_TOPICS.map((t) => (
                <option key={t} value={t}>
                  {capitalize(t)}
                </option>
              ))}
            </select>
          </label>
        )}
        {game === "story" && <DiceSetup rolls={rolls} onChange={setRolls} />}
        {duel && rivals.length > 0 && (
          <label className="block text-sm">
            <span className="mb-1 block font-semibold">Against</span>
            <select
              className={INPUT}
              value={opponent}
              onChange={(e) => setOpponent(e.target.value)}
            >
              <option value="">Nobody: practice alone</option>
              {rivals.map((r) => (
                <option key={r.productId} value={r.productId}>
                  {`Duel ${r.companyName}'s ${r.productName}`}
                </option>
              ))}
            </select>
          </label>
        )}
        {error && (
          <p role="alert" className="text-sm text-danger">
            {error}
          </p>
        )}
        <div className="flex flex-wrap gap-2">
          <Button
            onClick={() =>
              onPractice({ game, difficulty, settings, versionId, opponent: opponent || null })
            }
          >
            {opponent ? "Start the duel" : "Practice (free)"}
          </Button>
          {shop && !opponent && (
            <Button
              variant="secondary"
              disabled={showcaseReason !== null}
              title={showcaseReason ?? undefined}
              onClick={async () => {
                setError(null);
                const r = await socket.command(view.world.id, {
                  type: "startShowcase",
                  game,
                  difficulty,
                  versionId,
                  settings,
                });
                if (!r.ok) return setError(r.reason);
                await onShowcase();
              }}
            >
              {`Showcase live for the press (${formatMoney(shop.fee)})`}
            </Button>
          )}
        </div>
        {shop && !opponent && (
          <p className="text-xs text-muted">
            {showcaseReason ??
              "Practice is private. A showcase plays in front of the press: a great run wins fans, a bad one gets roasted."}
          </p>
        )}
      </div>
    </Panel>
  );
}

function score(steps: { correct: boolean; points: number | null }[], game: GameId): number {
  return game === "story"
    ? steps.reduce((a, s) => a + (s.points ?? 0), 0)
    : depthOf(steps as never);
}

function PlayView({
  view,
  game,
  difficulty,
  run,
  opponent,
  best,
  onBack,
}: {
  view: WorldView;
  game: GameId;
  difficulty: Difficulty;
  run: LiveRun;
  /** The rival product a duel is against. */
  opponent: string | null;
  best: number | null;
  onBack: () => void;
}) {
  const Stage = STAGES[game];
  const me = view.companies.find((c) => c.id === view.me.company?.id);
  const color = me?.palette.primary ?? "var(--brand)";
  const duel = run.steps.b.length > 0 || Boolean(run.result?.sides.b);
  const typing = (side: "a" | "b") =>
    Object.fromEntries(
      Object.entries(run.typing)
        .filter(([k]) => k.startsWith(`${side}:`))
        .map(([k, v]) => [Number(k.split(":")[1]), v]),
    );
  const mine = score(run.steps.a, game);
  const theirs = score(run.steps.b, game);
  const max = game === "story" ? 39 : GAME_STEPS[game];
  const them = rivalAis(view).find((r) => r.productId === opponent);
  const rival = view.companies.find((c) => c.id === them?.companyId);
  return (
    <Panel
      title={`${GAME_INFO[game].icon} ${GAME_NAMES[game]} · ${capitalize(difficulty)}`}
      actions={
        <Button variant="secondary" onClick={onBack}>
          {run.status === "running" || run.status === "starting" ? "Leave" : "Play another"}
        </Button>
      }
    >
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <span data-testid="score" className="font-semibold">{`${Math.round(mine)} of ${max}`}</span>
        {game !== "story" && <Lives steps={run.steps.a} />}
        {run.status === "starting" && <span className="text-muted">Getting your AI ready…</span>}
        {run.status === "running" && <span className="text-muted">Playing…</span>}
        {best !== null && <span className="text-muted">{`Your best: ${Math.round(best)}`}</span>}
      </div>
      {run.error && (
        <p role="alert" className="mb-3 text-sm text-danger">
          {run.error}
        </p>
      )}
      {duel ? (
        <div className="grid gap-4 md:grid-cols-2">
          {(["a", "b"] as const).map((side) => (
            <section key={side} aria-label={side === "a" ? "Your AI" : "Their AI"}>
              <p
                className="mb-2 text-xs font-semibold uppercase tracking-wider"
                style={{ color: side === "a" ? color : rival?.palette.primary }}
              >
                {side === "a"
                  ? `${me?.name ?? "You"} · ${Math.round(mine)}`
                  : `${them ? `${them.companyName}'s ${them.productName}` : "Rival"} · ${Math.round(theirs)}`}
              </p>
              <Stage
                steps={run.steps[side]}
                typing={typing(side)}
                color={side === "a" ? color : (rival?.palette.primary ?? "#888")}
                compact
              />
            </section>
          ))}
        </div>
      ) : (
        <Stage steps={run.steps.a} typing={typing("a")} color={color} />
      )}
      {run.status === "done" && (
        <motion.p
          initial={{ scale: 1.3, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="mt-4 font-display text-xl font-bold"
          data-testid="result"
        >
          {duel
            ? mine > theirs
              ? `You win the duel, ${Math.round(mine)}–${Math.round(theirs)}!`
              : mine < theirs
                ? `They win, ${Math.round(theirs)}–${Math.round(mine)}.`
                : `A tie, ${Math.round(mine)}–${Math.round(theirs)}.`
            : best !== null && mine >= best
              ? `New personal best: ${Math.round(mine)}!`
              : `Finished with ${Math.round(mine)} of ${max}.`}
        </motion.p>
      )}
    </Panel>
  );
}
