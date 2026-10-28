import { PLAN_IDS, type PlanId, TEXT_LIMITS, type WorldView } from "@airace/protocol";
import { getBalance } from "@airace/sim";
import { useEffect, useId, useRef, useState } from "react";
import { Button } from "../components/Button.tsx";
import { INPUT } from "../components/Field.tsx";
import { Panel } from "../components/Panel.tsx";
import { formatMoney } from "../format.ts";
import { useSocket } from "../store/session.ts";

/** Why the newest AI can't be launched yet, or null when it can. */
export function launchBlocked(view: WorldView): string | null {
  const latest = view.me.company?.versions.at(-1);
  if (!latest) return "Train your first AI before launching it.";
  if (latest.scores === null) return "Wait until your AI has been scored before launching it.";
  return null;
}

/** What each plan is, in plain words, with today's price. */
export function planWords(view: WorldView, plan: PlanId): { name: string; what: string } {
  const plans = view.me.company?.plans;
  switch (plan) {
    case "free":
      return {
        name: "Free",
        what: "No price. Brings the most people; they cost you serving but don't pay.",
      };
    case "personal":
      return {
        name: "Personal",
        what: `${formatMoney(plans?.personal.price ?? 0)} a month per person. Students and everyday users leave as the price rises.`,
      };
    case "business":
      return {
        name: "Business",
        what: `${formatMoney(plans?.business.price ?? 0)} a month per seat. Businesses, developers and researchers buy several seats and mind the price less.`,
      };
  }
}

/**
 * Launch day (game-layer design §6): pick a scored AI, name the product, choose
 * the plans it will power and a countdown. Hype builds until it goes on sale.
 */
export function LaunchDialog({
  view,
  versionId,
  onClose,
}: {
  view: WorldView;
  versionId?: string;
  onClose: () => void;
}) {
  const socket = useSocket();
  const scored = [...(view.me.company?.versions ?? [])].reverse().filter((v) => v.scores !== null);
  const { countdown } = getBalance(view.world.balance).economy;
  const [min, max] = TEXT_LIMITS.productName;
  const [version, setVersion] = useState(versionId ?? scored[0]?.id ?? "");
  const [name, setName] = useState("");
  const [plans, setPlans] = useState<PlanId[]>(["free", "personal"]);
  const [days, setDays] = useState(Math.min(countdown.max, Math.max(countdown.min, 5)));
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);
  const titleId = useId();
  const nameId = useId();

  useEffect(() => nameRef.current?.focus(), []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const trimmed = name.trim();
  const nameOk = trimmed.length >= min && trimmed.length <= max;
  const toggle = (plan: PlanId) =>
    setPlans((ps) => (ps.includes(plan) ? ps.filter((p) => p !== plan) : [...ps, plan]));

  async function launch() {
    setBusy(true);
    setError(null);
    const r = await socket.command(view.world.id, {
      type: "launchProduct",
      versionId: version,
      name: trimmed,
      plans: PLAN_IDS.filter((p) => plans.includes(p)),
      countdownDays: days,
    });
    setBusy(false);
    if (r.ok) onClose();
    else setError(r.reason);
  }

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4 backdrop-blur-sm">
      <Panel
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="w-full max-w-lg"
        title={<span id={titleId}>Launch an AI</span>}
        hint="Announce it, build hype during the countdown, and it goes on sale on launch day."
      >
        <div className="space-y-4">
          <label className="block text-sm">
            <span className="font-semibold">Which AI</span>
            <select
              className={`${INPUT} mt-1`}
              value={version}
              onChange={(e) => setVersion(e.target.value)}
            >
              {scored.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label} · overall score {Math.round(v.overall ?? 0)}
                </option>
              ))}
            </select>
          </label>
          <div className="text-sm">
            <label htmlFor={nameId} className="font-semibold">
              Product name
            </label>
            <input
              id={nameId}
              ref={nameRef}
              className={`${INPUT} mt-1`}
              placeholder="Spark"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <p className={`mt-1 text-xs ${name && !nameOk ? "text-danger" : "text-muted"}`}>
              {min}–{max} characters. Customers see this name.
            </p>
          </div>
          <fieldset className="m-0 space-y-2 border-0 p-0 text-sm">
            <legend className="mb-1 font-semibold">Plans it will power</legend>
            {PLAN_IDS.map((plan) => {
              const words = planWords(view, plan);
              return (
                <label key={plan} className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    className="mt-1 accent-[var(--brand)]"
                    checked={plans.includes(plan)}
                    onChange={() => toggle(plan)}
                  />
                  <span>
                    <span className="font-semibold">{words.name}</span>
                    <span className="block text-xs text-muted">{words.what}</span>
                  </span>
                </label>
              );
            })}
          </fieldset>
          <label className="block text-sm">
            <span className="font-semibold">Countdown: {days} game days</span>
            <input
              type="range"
              className="mt-2 w-full accent-[var(--brand)]"
              min={countdown.min}
              max={countdown.max}
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
            />
            <span className="text-xs text-muted">A longer countdown builds more hype.</span>
          </label>
          {error && (
            <p role="alert" className="text-sm text-danger">
              {error}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={() => void launch()} disabled={!nameOk || !version || busy}>
              Launch
            </Button>
          </div>
        </div>
      </Panel>
    </div>
  );
}
