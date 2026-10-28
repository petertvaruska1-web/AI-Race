import {
  GROUP_IDS,
  type MonthLedgerView,
  PLAN_IDS,
  type PlanId,
  type WorldView,
} from "@airace/protocol";
import { getBalance } from "@airace/sim";
import { useState } from "react";
import { Button } from "../components/Button.tsx";
import { INPUT } from "../components/Field.tsx";
import { Panel } from "../components/Panel.tsx";
import { Stat } from "../components/Stat.tsx";
import { formatMoney } from "../format.ts";
import { GROUPS } from "../groups.ts";
import { useSocket } from "../store/session.ts";
import { LaunchDialog, launchBlocked, planWords } from "./LaunchDialog.tsx";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const total = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);
const monthName = (m: MonthLedgerView) => MONTHS[Number(m.month.slice(5)) - 1] ?? m.month;

/** Labelled bars, each as long as its share of the largest. */
function Bars({ rows }: { rows: { label: string; value: number; color?: string }[] }) {
  const top = Math.max(1, ...rows.map((r) => Math.abs(r.value)));
  return (
    <ul className="space-y-2 text-sm">
      {rows.map((r) => (
        <li key={r.label}>
          <div className="flex justify-between gap-3">
            <span className="text-muted">{r.label}</span>
            <span className="tabular font-semibold">{formatMoney(r.value)}</span>
          </div>
          <div className="mt-1 h-1.5 rounded-full bg-surface-2">
            <div
              className="h-1.5 rounded-full"
              style={{
                width: `${(100 * Math.abs(r.value)) / top}%`,
                background: r.color ?? "var(--brand)",
              }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

/** Where money comes from and goes, the AIs on sale, and the plans and prices (game-layer §8). */
export function Money({ view }: { view: WorldView }) {
  const socket = useSocket();
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const company = view.me.company;
  if (!company) return null;
  const now = company.ledger.current;
  const last = company.ledger.months.at(-1);
  const income = total(now.income);
  const costs = total(now.costs);

  async function send(cmd: Parameters<typeof socket.command>[1]) {
    setError(null);
    const r = await socket.command(view.world.id, cmd);
    if (!r.ok) setError(r.reason);
  }

  return (
    <div className="space-y-5">
      {error && (
        <p role="alert" className="panel border-danger px-4 py-3 text-sm text-danger">
          {error}
        </p>
      )}
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Coming in" aria-label="Coming in" hint="What customers paid this month.">
          <Stat
            label="This month"
            value={income}
            delta={last ? income - total(last.income) : undefined}
            deltaLabel="against last month"
            format={formatMoney}
          />
          <h3 className="mt-5 mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
            By plan
          </h3>
          <Bars
            rows={[
              { label: "Personal", value: now.income.personal },
              { label: "Business", value: now.income.business, color: "var(--brand-2)" },
            ]}
          />
          <h3 className="mt-5 mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
            By customer group
          </h3>
          <Bars
            rows={GROUP_IDS.map((g) => ({ label: GROUPS[g].many, value: now.revenueByGroup[g] }))}
          />
        </Panel>

        <Panel title="Going out" hint="What running the company cost this month.">
          <Stat
            label="This month"
            value={costs}
            delta={last ? costs - total(last.costs) : undefined}
            deltaLabel="against last month"
            format={formatMoney}
          />
          <div className="mt-5">
            <Bars
              rows={[
                { label: "Training", value: now.costs.training, color: "var(--danger)" },
                { label: "Serving customers", value: now.costs.serving, color: "var(--danger)" },
                {
                  label: "Fact-checking data",
                  value: now.costs.dataChecks,
                  color: "var(--danger)",
                },
              ]}
            />
          </div>
          <p className="mt-4 text-xs text-muted">
            Serving costs grow with your customers, free ones too, and with the size of the AI
            behind their plan.
          </p>
        </Panel>
      </div>

      <Panel title="Profit by month" hint="Money in, less money out.">
        <ProfitBars months={[...company.ledger.months, now]} />
      </Panel>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel
          title="Your AIs on sale"
          hint="Up to three products at once. Each is one of your AIs under a name."
          actions={
            <Button
              onClick={() => setLaunching(true)}
              disabled={launchBlocked(view) !== null}
              title={launchBlocked(view) ?? undefined}
            >
              Launch an AI
            </Button>
          }
        >
          {company.products.length === 0 ? (
            <p className="text-sm text-muted">
              Nothing on sale yet. Launch an AI to get customers.
            </p>
          ) : (
            <ul className="divide-y divide-border">
              {company.products.map((p) => {
                const v = company.versions.find((x) => x.id === p.versionId);
                return (
                  <li
                    key={p.id}
                    className={`flex items-center gap-3 py-3 ${p.status === "retired" ? "opacity-50" : ""}`}
                  >
                    <div className="mr-auto">
                      <p className="player-text font-semibold">{p.name}</p>
                      <p className="text-xs text-muted">
                        {v?.label} ·{" "}
                        {p.status === "live"
                          ? "On sale"
                          : p.status === "countdown"
                            ? `Launches in ${p.launchDay - view.world.day} game days`
                            : "Retired"}
                      </p>
                    </div>
                    {p.status !== "retired" && (
                      <Button
                        variant="ghost"
                        aria-label={`Retire ${p.name}`}
                        title="Take it off sale. Move its plans to another AI first."
                        onClick={() => void send({ type: "retireProduct", productId: p.id })}
                      >
                        Retire
                      </Button>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </Panel>

        <Panel
          title="Plans and prices"
          hint="Which AI each plan sells, and what it costs customers."
        >
          <div className="space-y-5">
            {PLAN_IDS.map((plan) => (
              <PlanRow key={plan} view={view} plan={plan} send={send} />
            ))}
          </div>
        </Panel>
      </div>

      {launching && <LaunchDialog view={view} onClose={() => setLaunching(false)} />}
    </div>
  );
}

function PlanRow({
  view,
  plan,
  send,
}: {
  view: WorldView;
  plan: PlanId;
  send: (cmd: Parameters<ReturnType<typeof useSocket>["command"]>[1]) => Promise<void>;
}) {
  const company = view.me.company;
  const words = planWords(view, plan);
  const live = (company?.products ?? []).filter((p) => p.status === "live");
  const current = company?.plans[plan].productId ?? null;
  const paid = plan === "free" ? null : plan;
  const bounds = paid ? getBalance(view.world.balance).economy.prices[paid] : null;
  const [price, setPrice] = useState(String(paid ? (company?.plans[paid].price ?? "") : ""));
  const [problem, setProblem] = useState<string | null>(null);

  function setPriceNow() {
    if (!paid || !bounds) return;
    const n = Number(price);
    if (!Number.isInteger(n) || n < bounds.min || n > bounds.max) {
      setProblem(
        `Choose a whole-dollar price from ${formatMoney(bounds.min)}–${formatMoney(bounds.max)}.`,
      );
      return;
    }
    setProblem(null);
    void send({ type: "setPrice", plan: paid, price: n });
  }

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <p className="w-20 font-semibold">{words.name}</p>
        <select
          aria-label={`AI for the ${words.name.toLowerCase()} plan`}
          className={`${INPUT} w-auto flex-1`}
          value={current ?? ""}
          onChange={(e) => void send({ type: "setPlan", plan, productId: e.target.value || null })}
        >
          <option value="">Closed</option>
          {live.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>
      <p className="mt-1 text-xs text-muted">{words.what}</p>
      {paid && bounds && (
        <div className="mt-2 flex items-center gap-2">
          <span className="text-sm text-muted">$</span>
          <input
            aria-label={`${words.name} price`}
            className={`${INPUT} w-24`}
            inputMode="numeric"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
          />
          <span className="text-xs text-muted">
            a month{paid === "business" ? " per seat" : ""} · {formatMoney(bounds.min)}–
            {formatMoney(bounds.max)}
          </span>
          <Button
            variant="secondary"
            aria-label={`Set the ${paid} price`}
            onClick={setPriceNow}
            className="ml-auto"
          >
            Set
          </Button>
        </div>
      )}
      {problem && (
        <p role="alert" className="mt-1 text-xs text-danger">
          {problem}
        </p>
      )}
    </div>
  );
}

/** Each month's profit as a bar above or below the zero line. */
function ProfitBars({ months }: { months: MonthLedgerView[] }) {
  const profits = months.map((m) => total(m.income) - total(m.costs));
  const top = Math.max(1, ...profits.map(Math.abs));
  return (
    <div className="flex h-40 items-stretch gap-2" role="img" aria-label="Profit by month">
      {months.map((m, i) => {
        const p = profits[i] ?? 0;
        const h = `${(100 * Math.abs(p)) / top}%`;
        return (
          <div key={m.month} className="flex min-w-8 flex-1 flex-col items-center">
            <div className="flex w-full flex-1 flex-col justify-end">
              {p > 0 && <div className="rounded-t bg-good/80" style={{ height: h }} />}
            </div>
            <div className="h-px w-full bg-border" />
            <div className="flex w-full flex-1 flex-col">
              {p < 0 && <div className="rounded-b bg-danger/80" style={{ height: h }} />}
            </div>
            <span className="mt-1 text-[11px] text-muted">{monthName(m)}</span>
          </div>
        );
      })}
    </div>
  );
}
