import type { WorldView } from "@airace/protocol";
import { type Balance, getBalance, paramCount } from "@airace/sim";
import { useEffect, useState } from "react";
import { Panel } from "../components/Panel.tsx";
import { api } from "../net/api.ts";
import {
  type CardView,
  shortCardName,
  type TrainingLimit,
  usableCards,
  useCards,
} from "../store/cards.ts";
import { useWindows } from "../store/windows.ts";
import { WindowFrame, type WindowProps } from "../windows/WindowFrame.tsx";

export const NO_CARD = "Connect your computer to do this.";

const named = (id: string) => id.charAt(0).toUpperCase() + id.slice(1);

/** What the Lab may train on the cards that work for this player (Stage 2). */
export type LabLimit = {
  /** No card of theirs (Fair off) or no card at all (Fair on) is online. */
  noCard: boolean;
  /** The biggest brain's parameters, or null while unknown. */
  maxParams: number | null;
  /** Why bigger brains are off, in plain words. */
  note: string | null;
  /** Training tokens a second on the reference model (0: unknown). */
  tokensPerSecond: number;
  /** "on your card" or "on the shared cards". */
  where: string;
};

export function labLimit(
  view: WorldView,
  cards: CardView[] | null,
  limit: TrainingLimit | null,
  balance: Balance = getBalance(view.world.balance),
): LabLimit {
  const fair = view.world.fairMode;
  const where = fair ? "on the shared cards" : "on your card";
  if (!cards) return { noCard: false, maxParams: null, note: null, tokensPerSecond: 0, where };
  const noCard = usableCards(cards, view.me.userId, fair).length === 0;
  if (noCard || !limit) {
    return { noCard, maxParams: null, note: null, tokensPerSecond: 0, where };
  }
  const brain = balance.brainSizes.find((b) => b.id === limit.maxBrain) ?? null;
  const who = fair ? "The shared cards" : "Your graphics card";
  return {
    noCard,
    maxParams: brain ? paramCount(brain.nLayer, brain.dModel) : 0,
    note: brain
      ? `${who} can train up to the ${named(brain.id)} brain.`
      : `${who} ${fair ? "are" : "is"} too small to train an AI.`,
    tokensPerSecond: limit.tokensPerSecond,
    where,
  };
}

/** What an RTX 3050 measures on the helper's speed test (the server's `SPEED_REFERENCE`). */
export const REFERENCE_TOKENS_PER_SECOND = 450_000;
/**
 * Microseconds a training token costs on that card, measured with training engine 2 (ADR 0009):
 * a fixed part plus a part that grows a little slower than the brain. Measured: 4.4 for the
 * Tiny brain (1.4M parameters), 10.5 for Medium (7.5M), 21.4 for Large (19.3M), 37.4 for Huge
 * (40.7M); this gives 4.3, 10.2, 20.6 and 37.7.
 */
const microsPerToken = (params: number) => 2.6 + 1.25 * (params / 1e6) ** 0.9;

/**
 * Real minutes a run of `tokens` takes on a card of this measured speed: the reference card's
 * measured cost of a token for this brain, scaled by how the card compares on the speed test.
 * (The old estimate took speed as falling in step with the brain's size from the tiny test
 * model, which made a mid-sized brain look almost four times slower than it trains.) Null when
 * the speed isn't known.
 */
export function realMinutes(
  tokens: number,
  params: number,
  tokensPerSecond: number,
): number | null {
  if (!(tokensPerSecond > 0) || !(params > 0)) return null;
  const seconds =
    (tokens * microsPerToken(params) * REFERENCE_TOKENS_PER_SECOND) / tokensPerSecond / 1e6;
  return Math.max(1, Math.round(seconds / 60));
}

/** The top bar's chip: your card, or the way to connect one. */
export function CardsChip({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const cards = useCards((s) => s.cards);
  const mine = (cards ?? []).find((c) => c.userId === view.me.userId && c.online);
  const label = mine
    ? `Your computer: ${shortCardName(mine.gpuName)} · online`
    : "Connect your computer";
  return (
    <button
      type="button"
      aria-label={label}
      onClick={(e) => open({ kind: "cards" }, e.currentTarget)}
      className={`rounded-full border px-3 py-1 text-xs font-semibold ${mine ? "border-border text-muted" : "border-brand text-brand"}`}
    >
      {label}
    </button>
  );
}

/** Cards with keys from what they are (a player may have two alike: the second is "#2"). */
function keyed(cards: CardView[]): { key: string; card: CardView }[] {
  const seen = new Map<string, number>();
  return cards.map((card) => {
    const base = `${card.userId}|${card.gpuName}|${card.house}|${card.online}`;
    const n = (seen.get(base) ?? 0) + 1;
    seen.set(base, n);
    return { key: `${base}#${n}`, card };
  });
}

function memory(vramMb: number): string {
  return vramMb > 0 ? `${Math.round(vramMb / 1024)} GB` : "";
}

/** Every player's cards, Fair mode's shared count, and your own helper's buttons. */
export function CardsPanel({ view }: { view: WorldView }) {
  const cards = useCards((s) => s.cards) ?? [];
  const [key, setKey] = useState<string | null>(null);
  useEffect(() => {
    api
      .helperKey()
      .then(setKey)
      .catch(() => setKey(null));
  }, []);
  const online = cards.filter((c) => c.online);
  const connect = key
    ? `airace-helper://connect?url=${encodeURIComponent(location.origin)}&key=${encodeURIComponent(key)}`
    : null;
  return (
    <div className="space-y-4">
      {view.world.fairMode && (
        <p className="text-sm">
          <span className="font-semibold">{`Shared cards: ${online.length} online`}</span>
          <span className="text-muted"> · each company gets an equal share.</span>
        </p>
      )}
      <ul className="divide-y divide-border">
        {keyed(cards).map(({ key, card: c }) => (
          <li key={key} className="flex items-center gap-3 py-2">
            <span
              aria-hidden
              className={`inline-block h-2 w-2 rounded-full ${c.online ? "bg-good" : "bg-border"}`}
            />
            <span className="player-text font-semibold">{c.playerName}</span>
            <span className="text-sm text-muted">
              {c.online
                ? [
                    shortCardName(c.gpuName),
                    memory(c.vramMb),
                    c.speed > 0 ? `${c.speed}× speed` : "",
                    c.house ? "runs the game" : "",
                    c.outdated ? "needs an update: run the helper's setup file again" : "",
                  ]
                    .filter(Boolean)
                    .join(" · ")
                : "not connected"}
            </span>
          </li>
        ))}
      </ul>
      <Panel title="Your computer" hint="The helper lets this PC train and run your company's AIs.">
        <ol className="mb-3 list-decimal space-y-1 pl-5 text-sm text-muted">
          <li>Get the helper and run it once (about 15 minutes the first time).</li>
          <li>Then press Connect your computer whenever you play.</li>
        </ol>
        <div className="flex flex-wrap gap-3">
          <a className="rounded-lg border border-border px-3 py-2 text-sm" href="/helper/setup.cmd">
            Get the helper
          </a>
          {connect && (
            <a className="rounded-lg bg-brand px-3 py-2 text-sm text-on-brand" href={connect}>
              Connect your computer
            </a>
          )}
        </div>
      </Panel>
    </div>
  );
}

export function CardsWindow({ entry, view }: WindowProps) {
  return (
    <WindowFrame
      entry={entry}
      title="Graphics cards"
      subtitle={
        view.world.fairMode
          ? "Fair mode: everyone's cards are shared, an equal share for each company."
          : "Each player's AI trains on their own computer."
      }
    >
      <CardsPanel view={view} />
    </WindowFrame>
  );
}
