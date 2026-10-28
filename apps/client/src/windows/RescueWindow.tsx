import { getBalance } from "@airace/sim";
import { useState } from "react";
import { Button } from "../components/Button.tsx";
import { dayLabel, formatMoney } from "../format.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { WindowFrame, type WindowProps } from "./WindowFrame.tsx";

/** The two ways out when cash runs out (game-layer design §8, C19), with their price. */
export function RescueWindow({ entry, view }: WindowProps) {
  const socket = useSocket();
  const back = useWindows((s) => s.back);
  const [problem, setProblem] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const company = view.me.company;
  const pub = view.companies.find((c) => c.id === company?.id);
  const offer = company?.rescue;
  if (!company || !pub || !offer) {
    return (
      <WindowFrame entry={entry} title="Choose a rescue">
        <p className="text-sm text-muted">No rescue is waiting: your cash is fine.</p>
      </WindowFrame>
    );
  }
  const r = getBalance(view.world.balance).economy.rescue;
  const amount = Math.max(0, Math.round(r.cushion - company.cash));
  const owed = (company.loan?.balance ?? 0) + amount;
  const keep = Math.round(pub.ownership * (1 - r.investorStake) * 100);

  async function choose(kind: "loan" | "investor") {
    setProblem(null);
    setSending(true);
    const res = await socket.command(view.world.id, { type: "chooseRescue", kind });
    setSending(false);
    if (res.ok) back();
    else setProblem(res.reason);
  }

  return (
    <WindowFrame
      entry={entry}
      title="Choose a rescue"
      subtitle={`Your cash ran out. Choose by ${dayLabel(offer.deadlineDay)}, or the emergency loan is taken for you.`}
      wide
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <section className="rounded-2xl border border-border bg-surface-2/50 p-5">
          <h3 className="text-lg font-semibold">Emergency loan</h3>
          <p className="mt-1 text-sm text-muted">Keep all of your company, and pay it back.</p>
          <p className="mt-4 text-xs font-semibold uppercase tracking-wider text-muted">
            Cash back
          </p>
          <p className="font-display text-3xl font-bold">{formatMoney(amount)}</p>
          <ul className="mt-3 space-y-1 text-sm">
            <li>{`${Math.round(r.loanMonthlyRate * 100)}% interest a month on what you owe`}</li>
            <li>
              {`Repay ${formatMoney(Math.ceil(owed / r.loanTermMonths))} a month for up to ${r.loanTermMonths} months`}
            </li>
          </ul>
          <Button className="mt-5 w-full" disabled={sending} onClick={() => void choose("loan")}>
            Take the emergency loan
          </Button>
        </section>
        <section className="rounded-2xl border border-border bg-surface-2/50 p-5">
          <h3 className="text-lg font-semibold">Investor</h3>
          <p className="mt-1 text-sm text-muted">
            Nothing to pay back, but part of the company is theirs.
          </p>
          <p className="mt-4 text-xs font-semibold uppercase tracking-wider text-muted">
            Cash back
          </p>
          <p className="font-display text-3xl font-bold">{formatMoney(amount)}</p>
          <ul className="mt-3 space-y-1 text-sm">
            <li>{`They take ${Math.round(r.investorStake * 100)}% of the company`}</li>
            <li className="font-semibold">{`You keep ${keep}%`}</li>
          </ul>
          <Button
            variant="secondary"
            className="mt-5 w-full"
            disabled={sending}
            onClick={() => void choose("investor")}
          >
            Bring in an investor
          </Button>
        </section>
      </div>
      {problem && (
        <p role="alert" className="mt-4 text-sm text-danger">
          {problem}
        </p>
      )}
    </WindowFrame>
  );
}
