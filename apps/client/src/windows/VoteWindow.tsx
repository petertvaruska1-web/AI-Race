import { useEffect, useState } from "react";
import { Button } from "../components/Button.tsx";
import { useSocket } from "../store/session.ts";
import { useVotes } from "../store/votes.ts";
import { GoneWindow, WindowFrame, type WindowProps } from "./WindowFrame.tsx";

function timeLeft(ms: number): string {
  const s = Math.max(0, Math.ceil(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/**
 * A vote on two other players' story battle (Stage 2): each round's two stories, left and right,
 * with nothing saying whose they are. The stories are the AIs' real words, shown as text (R1).
 */
export function VoteWindow({ entry, view, battleId }: WindowProps & { battleId: string }) {
  const socket = useSocket();
  const ballot = useVotes((s) => s.ballots[battleId]);
  const picked = useVotes((s) => s.picked);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  if (!ballot) return <GoneWindow entry={entry} />;
  const open = !ballot.closed && ballot.closesAt > now;

  async function vote(round: number, pick: "left" | "right") {
    setError(null);
    const r = await socket.battleVote(view.world.id, battleId, round, pick);
    if (r.ok) picked(battleId, round, pick);
    else setError(r.reason);
  }

  return (
    <WindowFrame
      entry={entry}
      title="Which story is better?"
      subtitle="Two players' AIs wrote these. Your votes decide each round."
    >
      <p className="mb-4 text-sm text-muted">
        {open
          ? `${timeLeft(ballot.closesAt - now)} left to vote`
          : "Voting is over. Thanks for voting!"}
      </p>
      <div className="space-y-6">
        {ballot.rounds.map((r) => {
          const total = ballot.closed?.find((t) => t.round === r.round);
          return (
            <section key={r.round} data-testid={`vote-round-${r.round}`}>
              <p className="mb-2 text-sm font-semibold">{r.prompt}</p>
              <div className="grid gap-3 sm:grid-cols-2">
                {(["left", "right"] as const).map((side) => (
                  <div
                    key={side}
                    className={`rounded-xl border p-3 ${ballot.picks[r.round] === side ? "border-brand bg-brand-soft" : "border-border"}`}
                  >
                    <p className="mb-3 whitespace-pre-wrap text-sm">{r[side]}</p>
                    {total ? (
                      <p className="text-xs text-muted">
                        {total[side]} {total[side] === 1 ? "vote" : "votes"}
                      </p>
                    ) : (
                      <Button
                        variant={ballot.picks[r.round] === side ? "primary" : "secondary"}
                        disabled={!open}
                        onClick={() => void vote(r.round, side)}
                      >
                        This one
                      </Button>
                    )}
                  </div>
                ))}
              </div>
            </section>
          );
        })}
      </div>
      {error && (
        <p role="alert" className="mt-3 text-sm text-danger">
          {error}
        </p>
      )}
    </WindowFrame>
  );
}
