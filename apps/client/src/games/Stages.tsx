import {
  GAME_LIVES,
  GAME_STEPS,
  type GameId,
  type MinigameStep,
  STORY_MAX_POINTS,
} from "@airace/protocol";
import { AnimatePresence, motion } from "motion/react";
import { DICE, type Pic, Picture } from "./pictures.tsx";

// The game stages (playtest pack §4.2): each draws the steps the worker streamed, with the
// AI's own choice or words (R1). A stage never decides anything: it only shows what happened.

export type StageProps = {
  steps: MinigameStep[];
  /** What the AI is typing now, by attempt. */
  typing: Record<number, string>;
  /** The AI's color (its company's). */
  color: string;
  compact?: boolean;
};

const CHECK_WORDS: Record<string, string> = {
  one_word: "Answer in one word",
  yes_no: "Answer yes or no",
  list_n: "Give a list",
  repeat_word: "Repeat a word",
  starts_with: "Start with a given word",
  contains_any: "Mention the right thing",
  all_caps: "Answer in capital letters",
};

const last = <T,>(xs: T[]) => xs[xs.length - 1];
export const depthOf = (steps: MinigameStep[]) => steps.filter((s) => s.correct).length;

export function Lives({ steps }: { steps: MinigameStep[] }) {
  const left = last(steps)?.lives ?? GAME_LIVES;
  return (
    <span role="img" aria-label={`${left} lives left`} className="text-lg tracking-wider">
      {Array.from({ length: GAME_LIVES }, (_, k) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: one heart per life
        <span key={k} className={k < left ? "text-danger" : "opacity-25"}>
          ♥
        </span>
      ))}
    </span>
  );
}

/** How sure the AI was of each option, as bars (its own likelihoods; R1). */
function SureBars({
  step,
  color,
  labels,
}: {
  step: MinigameStep;
  color: string;
  labels?: string[];
}) {
  if (!step.options || !step.sure) return null;
  return (
    <ul className="mt-3 space-y-1.5 text-sm">
      {step.options.map((o, i) => {
        const chosen = step.choice === i;
        const right = step.answerIndex === i;
        return (
          <li key={o} className="grid grid-cols-[minmax(0,10rem)_1fr_3rem] items-center gap-2">
            <span className={`truncate ${chosen ? "font-semibold" : "text-muted"}`}>
              {labels?.[i] ?? o}
              {right && <span className="ml-1 text-good">✓</span>}
            </span>
            <span className="h-2 rounded-full bg-surface-2">
              <motion.span
                className="block h-2 rounded-full"
                style={{ backgroundColor: chosen ? color : "var(--color-muted)" }}
                initial={{ width: 0 }}
                animate={{ width: `${Math.round(100 * (step.sure?.[i] ?? 0))}%` }}
              />
            </span>
            <span className="text-right text-xs text-muted">
              {Math.round(100 * (step.sure?.[i] ?? 0))}%
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function Verdict({ step }: { step: MinigameStep }) {
  return (
    <motion.p
      initial={{ scale: 1.4, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className={`mt-2 text-sm font-bold ${step.correct ? "text-good" : "text-danger"}`}
    >
      {step.correct ? "Right!" : "Not this time"}
    </motion.p>
  );
}

function Typed({ step, typing }: { step: MinigameStep; typing: Record<number, string> }) {
  const text = step.reply ?? typing[step.attempt] ?? "";
  return (
    <p className="player-text mt-2 whitespace-pre-wrap break-words rounded-lg bg-surface-2 p-2 text-sm">
      {text || <span className="italic text-muted">(no answer)</span>}
    </p>
  );
}

// -- Puzzle Maze ------------------------------------------------------------------------------

const COLS = 5;
const CELL = 56;

function cellXY(d: number): { x: number; y: number } {
  const row = Math.floor(d / COLS);
  const col = row % 2 === 0 ? d % COLS : COLS - 1 - (d % COLS);
  return { x: col * CELL + CELL / 2, y: row * CELL + CELL / 2 };
}

export function MazeStage({ steps, typing, color, compact }: StageProps) {
  const total = GAME_STEPS.maze;
  const depth = depthOf(steps);
  const rows = Math.ceil(total / COLS);
  const misses = (d: number) => steps.filter((s) => s.depth === d && !s.correct).length;
  const now = last(steps);
  const orb = cellXY(Math.min(depth, total - 1));
  return (
    <div>
      <svg
        viewBox={`0 0 ${COLS * CELL} ${rows * CELL}`}
        className={compact ? "w-full max-w-xs" : "w-full max-w-md"}
        role="img"
        aria-label={`Puzzle maze, depth ${depth} of ${total}`}
        data-depth={depth}
      >
        {Array.from({ length: total }, (_, d) => {
          const { x, y } = cellXY(d);
          const cleared = d < depth;
          return (
            // biome-ignore lint/suspicious/noArrayIndexKey: maze cells are fixed positions
            <g key={d}>
              <rect
                x={x - CELL / 2 + 4}
                y={y - CELL / 2 + 4}
                width={CELL - 8}
                height={CELL - 8}
                rx={10}
                fill={cleared ? color : "var(--color-surface-2)"}
                fillOpacity={cleared ? 0.35 : 1}
                stroke={d === depth ? color : "transparent"}
                strokeWidth={2}
              />
              {Array.from({ length: misses(d) }, (_, k) => (
                <text
                  // biome-ignore lint/suspicious/noArrayIndexKey: one mark per miss
                  key={k}
                  x={x - CELL / 2 + 10 + k * 12}
                  y={y + CELL / 2 - 8}
                  fontSize={11}
                  fill="var(--color-danger)"
                >
                  ✗
                </text>
              ))}
            </g>
          );
        })}
        <text x={cellXY(total - 1).x - 8} y={cellXY(total - 1).y + 6} fontSize={18}>
          🏁
        </text>
        <motion.circle
          r={12}
          fill={color}
          animate={{ cx: orb.x, cy: orb.y }}
          transition={{ type: "spring", stiffness: 120, damping: 14 }}
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
      </svg>
      {now && !compact && (
        <div className="mt-3 rounded-xl border border-border p-3">
          <p className="player-text font-semibold">{String(now.display.question ?? now.prompt)}</p>
          {now.options ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {now.options.map((o, i) => (
                <span
                  key={o}
                  className={`rounded-t-2xl border-2 px-4 pb-1 pt-3 text-sm font-semibold ${
                    now.choice === i
                      ? now.correct
                        ? "border-good text-good"
                        : "border-danger text-danger"
                      : "border-border text-muted"
                  }`}
                >
                  🚪 {o}
                </span>
              ))}
            </div>
          ) : (
            <Typed step={now} typing={typing} />
          )}
          <SureBars step={now} color={color} />
          <Verdict step={now} />
        </div>
      )}
    </div>
  );
}

// -- Pattern Bridge ---------------------------------------------------------------------------

export function BridgeStage({ steps, typing, color, compact }: StageProps) {
  const total = GAME_STEPS.bridge;
  const depth = depthOf(steps);
  const now = last(steps);
  const terms = (now?.display.terms as Pic[] | undefined) ?? [];
  const options = (now?.display.options as Pic[] | undefined) ?? [];
  const W = 36;
  return (
    <div>
      <svg
        viewBox={`0 0 ${(total + 2) * W} 120`}
        className={compact ? "w-full max-w-xs" : "w-full"}
        role="img"
        aria-label={`Pattern bridge, ${depth} tiles of ${total}`}
      >
        <defs>
          <linearGradient id="canyon" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="var(--color-surface-2)" />
            <stop offset="1" stopColor="#000" stopOpacity="0.6" />
          </linearGradient>
        </defs>
        <rect x={W} y={70} width={total * W} height={50} fill="url(#canyon)" />
        <rect x={0} y={60} width={W} height={60} fill="var(--color-surface-2)" />
        <rect x={(total + 1) * W} y={60} width={W} height={60} fill="var(--color-surface-2)" />
        <text x={(total + 1) * W + 8} y={52} fontSize={18}>
          🏁
        </text>
        {Array.from({ length: depth }, (_, d) => (
          <motion.rect
            // biome-ignore lint/suspicious/noArrayIndexKey: one tile per cleared step
            key={d}
            data-testid="tile"
            x={W + d * W + 2}
            y={60}
            width={W - 4}
            height={12}
            rx={3}
            fill={color}
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 60, opacity: 1 }}
          />
        ))}
        <AnimatePresence>
          {now && !now.correct && (
            <motion.rect
              key={now.attempt}
              data-testid="crumbled"
              x={W + depth * W + 2}
              width={W - 4}
              height={12}
              rx={3}
              fill="var(--color-danger)"
              initial={{ y: 60, opacity: 1, rotate: 0 }}
              animate={{ y: 120, opacity: 0, rotate: 25 }}
              transition={{ duration: 0.9 }}
            />
          )}
        </AnimatePresence>
        <motion.circle
          r={10}
          cy={48}
          fill={color}
          animate={{ cx: W / 2 + depth * W + (depth > 0 ? W / 2 : 0) }}
          transition={{ type: "spring", stiffness: 140, damping: 12 }}
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
      </svg>
      {now && !compact && (
        <div className="mt-3 rounded-xl border border-border p-3">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted">
            What comes next?
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            {terms.map((t, i) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: pattern terms repeat
              <Picture key={i} p={t} />
            ))}
            <span className="text-2xl text-muted">?</span>
          </div>
          {now.options ? (
            <div className="mt-3 flex flex-wrap gap-3">
              {options.map((o, i) => (
                <span
                  key={`${o.value}-${now.options?.[i]}`}
                  className={`rounded-xl border-2 p-1 ${now.choice === i ? (now.correct ? "border-good" : "border-danger") : "border-transparent opacity-60"}`}
                >
                  <Picture p={o} size={36} />
                </span>
              ))}
            </div>
          ) : (
            <Typed step={now} typing={typing} />
          )}
          <SureBars step={now} color={color} />
          <Verdict step={now} />
        </div>
      )}
    </div>
  );
}

// -- Quiz Show --------------------------------------------------------------------------------

const LETTERS = ["A", "B", "C", "D"];

export function QuizStage({ steps, typing, color, compact }: StageProps) {
  const total = GAME_STEPS.quiz;
  const depth = depthOf(steps);
  const now = last(steps);
  const shown = (now?.display.shown as string[] | undefined) ?? now?.options ?? [];
  const struck = new Set((now?.display.struck as string[] | undefined) ?? []);
  const chosen = now?.choice !== null && now?.options ? now.options[now.choice ?? -1] : null;
  const right =
    now?.options && now.answerIndex !== null
      ? now.options[now.answerIndex]
      : (now?.display.answer as string | undefined);
  return (
    <div className={`grid gap-4 ${compact ? "" : "md:grid-cols-[1fr_9rem]"}`}>
      {now && (
        <div className="rounded-2xl border border-border bg-gradient-to-b from-surface-2 to-transparent p-4">
          <p className="text-xs font-semibold uppercase tracking-wider" style={{ color }}>
            ★ Quiz Show · {String(now.display.topic ?? "")}
          </p>
          <p className="player-text mt-2 font-display text-lg font-semibold">
            {String(now.display.question ?? now.prompt)}
          </p>
          {now.options ? (
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {shown.map((o, i) => (
                <motion.div
                  key={o}
                  data-testid={struck.has(o) ? "struck" : "answer"}
                  animate={o === chosen ? { scale: [1, 1.05, 1] } : {}}
                  className={`rounded-xl border-2 px-3 py-2 text-sm ${
                    struck.has(o)
                      ? "border-transparent line-through opacity-30"
                      : o === chosen
                        ? now.correct
                          ? "border-good bg-good/10"
                          : "border-danger bg-danger/10"
                        : o === right
                          ? "border-good/60"
                          : "border-border"
                  }`}
                >
                  <span className="mr-2 font-bold text-muted">{LETTERS[i]}</span>
                  {o}
                </motion.div>
              ))}
            </div>
          ) : (
            <Typed step={now} typing={typing} />
          )}
          {!compact && <SureBars step={now} color={color} />}
          <Verdict step={now} />
        </div>
      )}
      {!compact && (
        <ol aria-label="Prize ladder" className="flex flex-col-reverse gap-1 text-xs">
          {Array.from({ length: total }, (_, k) => (
            <li
              // biome-ignore lint/suspicious/noArrayIndexKey: ladder rungs are fixed
              key={k}
              data-testid={k < depth ? "rung-won" : "rung"}
              className={`rounded px-2 py-0.5 text-right ${k < depth ? "font-bold text-on-brand" : k === depth ? "border border-border" : "text-muted"}`}
              style={k < depth ? { backgroundColor: color } : {}}
            >
              {k + 1}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

// -- Story Dice -------------------------------------------------------------------------------

function Die({ emoji, label, rolling }: { emoji: string; label: string; rolling: boolean }) {
  return (
    <motion.div
      className="flex w-24 flex-col items-center rounded-2xl border-2 border-border bg-surface-2 p-2 text-center"
      initial={rolling ? { rotate: -180, scale: 0.6 } : false}
      animate={{ rotate: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 160, damping: 12 }}
    >
      <span className="text-3xl">{emoji}</span>
      <span className="mt-1 text-[11px] leading-tight text-muted">{label}</span>
    </motion.div>
  );
}

export function StoryDiceStage({ steps, typing, color, compact }: StageProps) {
  const points = steps.reduce((a, s) => a + (s.points ?? 0), 0);
  return (
    <div className="space-y-4">
      <p className="text-sm font-semibold">
        {`${Math.round(points)} of ${STORY_MAX_POINTS} audience points`}
      </p>
      {steps.map((s) => {
        const faces = (s.display.faces as (number | null)[] | undefined) ?? [];
        const hero = DICE.hero[faces[0] ?? 0];
        const situation = DICE.situation[faces[1] ?? 0];
        const word = faces[2] === null || faces[2] === undefined ? null : DICE.word[faces[2]];
        const audience = s.audience ?? 0;
        return (
          <div key={s.attempt} className="rounded-2xl border border-border p-3">
            <div className="flex flex-wrap gap-2">
              {hero && <Die emoji={hero.emoji} label={hero.label} rolling />}
              {situation && <Die emoji={situation.emoji} label={situation.label} rolling />}
              {word ? (
                <Die emoji={word.emoji} label={`use "${word.label}"`} rolling />
              ) : (
                <Die emoji="🎲" label="no rule" rolling={false} />
              )}
            </div>
            {!compact && <Typed step={s} typing={typing} />}
            <div className="mt-2 flex items-center gap-3 text-sm">
              <span aria-hidden>👏</span>
              <span className="h-2 flex-1 rounded-full bg-surface-2">
                <motion.span
                  className="block h-2 rounded-full"
                  style={{ backgroundColor: color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.round(100 * audience)}%` }}
                />
              </span>
              {s.ruleMet !== null && (
                <span
                  data-testid="rule"
                  className={`rounded-full px-2 py-0.5 text-xs font-semibold ${s.ruleMet ? "bg-good/20 text-good" : "bg-danger/20 text-danger"}`}
                >
                  {s.ruleMet ? `used "${word?.label}" ✓` : `forgot "${word?.label}"`}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// -- Simon Says -------------------------------------------------------------------------------

export function SimonStage({ steps, typing, compact }: StageProps) {
  return (
    <ol className="space-y-2">
      <AnimatePresence initial={false}>
        {[...steps].reverse().map((s) => (
          <motion.li
            key={s.attempt}
            data-testid={s.correct ? "card-green" : "card-red"}
            initial={{ rotateX: 90, opacity: 0 }}
            animate={{ rotateX: 0, opacity: 1 }}
            className={`rounded-xl border-2 p-3 ${s.correct ? "border-good/70" : "border-danger/70"}`}
          >
            <p className="text-xs font-semibold uppercase tracking-wider text-muted">
              Simon says · {CHECK_WORDS[String(s.display.check)] ?? "Follow the instruction"}
            </p>
            <p className="player-text mt-1 font-semibold">
              {String(s.display.instruction ?? s.prompt)}
            </p>
            {!compact && <Typed step={s} typing={typing} />}
            <p className={`mt-1 text-sm font-bold ${s.correct ? "text-good" : "text-danger"}`}>
              {s.correct ? "Done right" : "Not quite"}
            </p>
          </motion.li>
        ))}
      </AnimatePresence>
    </ol>
  );
}

export const STAGES: Record<GameId, (p: StageProps) => React.ReactNode> = {
  maze: MazeStage,
  bridge: BridgeStage,
  quiz: QuizStage,
  story: StoryDiceStage,
  simon: SimonStage,
};
