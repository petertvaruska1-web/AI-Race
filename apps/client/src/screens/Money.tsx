import {
  COST_CATEGORIES,
  GROUP_IDS,
  INCOME_CATEGORIES,
  PLAN_IDS,
  type PlanId,
  REGION_IDS,
  REGION_NAMES,
  type WorldView,
} from "@airace/protocol";
import { getBalance } from "@airace/sim";
import { useState } from "react";
import { Bars } from "../components/Bars.tsx";
import { Button } from "../components/Button.tsx";
import { Computers } from "../components/Computers.tsx";
import { MonthBars } from "../components/charts/MonthBars.tsx";
import { INPUT } from "../components/Field.tsx";
import { Panel } from "../components/Panel.tsx";
import { Stat } from "../components/Stat.tsx";
import { formatMoney } from "../format.ts";
import { GROUPS } from "../groups.ts";
import { COST_NAMES, INCOME_NAMES, profitOf, total } from "../money.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { LaunchDialog, launchBlocked, planWords } from "./LaunchDialog.tsx";

/** Where money comes from and goes, the AIs on sale, and the plans and prices (game-layer §8). */
export function Money({ view }: { view: WorldView }) {
  const socket = useSocket();
  const open = useWindows((s) => s.open);
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
            rows={INCOME_CATEGORIES.map((c) => ({
              label: INCOME_NAMES[c],
              value: now.income[c],
              color: c === "business" ? "var(--brand-2)" : undefined,
              testId: `income-${c}`,
              onOpen: (el) => open({ kind: "income", category: c }, el),
            }))}
          />
          <h3 className="mt-5 mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
            By customer group
          </h3>
          <Bars
            rows={GROUP_IDS.map((g) => ({
              label: GROUPS[g].many,
              value: now.revenueByGroup[g],
              onOpen: (el) => open({ kind: "group", group: g }, el),
            }))}
          />
          <h3 className="mt-5 mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
            By region
          </h3>
          <Bars
            rows={REGION_IDS.map((r) => ({
              label: REGION_NAMES[r],
              value: now.revenueByRegion[r],
              color: "var(--brand-2)",
              onOpen: (el) => open({ kind: "region", region: r }, el),
            }))}
          />
        </Panel>

        <Panel
          title="Going out"
          aria-label="Going out"
          hint="What running the company cost this month."
        >
          <Stat
            label="This month"
            value={costs}
            delta={last ? costs - total(last.costs) : undefined}
            deltaLabel="against last month"
            format={formatMoney}
            downIsGood
          />
          <div className="mt-5">
            <Bars
              rows={COST_CATEGORIES.map((c) => ({
                label: COST_NAMES[c],
                value: now.costs[c],
                color: "var(--danger)",
                onOpen: (el) => open({ kind: "cost", category: c }, el),
              }))}
            />
          </div>
          <p className="mt-4 text-xs text-muted">
            Serving costs grow with your customers, free ones too, and with the size of the AI
            behind their plan.
          </p>
        </Panel>
      </div>

      {(now.financing.loans > 0 || now.financing.investors > 0 || now.financing.repaid > 0) && (
        <Panel
          title="Loans and investors"
          aria-label="Loans and investors"
          hint="Money that is neither income nor a cost: rescues in, repayments out."
        >
          <Bars
            rows={[
              { label: "Loans", value: now.financing.loans, color: "var(--good)" },
              { label: "Investors", value: now.financing.investors, color: "var(--good)" },
              { label: "Repaid", value: now.financing.repaid, color: "var(--danger)" },
            ].filter((r) => r.value > 0)}
          />
        </Panel>
      )}

      <Panel
        title="Computers"
        aria-label="Computers"
        hint="Where your AIs run: your data centers first, then rented cloud."
      >
        <Computers view={view} />
      </Panel>

      <Panel
        title="Profit by month"
        hint="Money in, less money out. Click a month for its report."
        actions={
          <button
            type="button"
            aria-label="Open money"
            title="Cash day by day"
            className="icon-btn"
            onClick={(e) => open({ kind: "money" }, e.currentTarget)}
          >
            ⤢
          </button>
        }
      >
        <MonthBars
          label="Profit by month"
          bars={[...company.ledger.months, now].map((m) => ({
            month: m.month,
            value: profitOf(m),
          }))}
          format={formatMoney}
          onOpen={(month, el) => open({ kind: "month", month }, el)}
        />
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
                    <button
                      type="button"
                      aria-label={`Open ${p.name}`}
                      onClick={(e) => open({ kind: "product", productId: p.id }, e.currentTarget)}
                      className="lift-row mr-auto rounded-md px-1 text-left"
                    >
                      <span className="player-text block font-semibold">{p.name}</span>
                      <span className="block text-xs text-muted">
                        {v?.label} ·{" "}
                        {p.status === "live"
                          ? "On sale"
                          : p.status === "countdown"
                            ? `Launches in ${p.launchDay - view.world.day} game days`
                            : "Retired"}
                      </span>
                    </button>
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
