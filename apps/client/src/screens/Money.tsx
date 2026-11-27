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
import { Display } from "../bench/Display.tsx";
import { KeyRow } from "../bench/KeyRow.tsx";
import { Knob } from "../bench/Knob.tsx";
import { Lamp } from "../bench/Lamp.tsx";
import { Module } from "../bench/Module.tsx";
import { PushButton } from "../bench/PushButton.tsx";
import { StampCard } from "../bench/StampCard.tsx";
import { Bars } from "../components/Bars.tsx";
import { Computers } from "../components/Computers.tsx";
import { MonthBars } from "../components/charts/MonthBars.tsx";
import { Stat } from "../components/Stat.tsx";
import { formatMoney, monthName } from "../format.ts";
import { GROUPS } from "../groups.ts";
import { COST_NAMES, INCOME_NAMES, profitOf, total } from "../money.ts";
import { FundPanel } from "../spending/Spending.tsx";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { gradeColor } from "../windows/ReportWindow.tsx";
import { LaunchDialog, launchBlocked, planWords } from "./LaunchDialog.tsx";

type Send = (cmd: Parameters<ReturnType<typeof useSocket>["command"]>[1]) => Promise<void>;

/**
 * The Money station as a bench (design §10, game-layer §8): Coming in and Going out as ledgers
 * in displays, each line a key that opens its window, with the loans and the report cards on a rack
 * under the costs; profit by month, the computers and the AIs on sale; and the plans and prices
 * with the fund.
 */
export function Money({ view }: { view: WorldView }) {
  const socket = useSocket();
  const open = useWindows((s) => s.open);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const company = view.me.company;
  const pub = view.companies.find((c) => c.id === company?.id);
  if (!company) return null;
  const now = company.ledger.current;
  const last = company.ledger.months.at(-1);
  // Other income shows once there is some, this month or last (royalties land at the close).
  const otherIncome = (c: (typeof INCOME_CATEGORIES)[number]) =>
    now.income[c] > 0 || (last?.income[c] ?? 0) > 0;
  const others = INCOME_CATEGORIES.filter((c) => !used(c, "plan") && otherIncome(c));
  const income = total(now.income);
  const costs = total(now.costs);
  const ownership = pub?.ownership ?? 1;
  const financed =
    now.financing.loans > 0 ||
    now.financing.investors > 0 ||
    now.financing.repaid > 0 ||
    company.loan !== null ||
    ownership < 1;
  const blocked = launchBlocked(view);

  const send: Send = async (cmd) => {
    setError(null);
    const r = await socket.command(view.world.id, cmd);
    if (!r.ok) setError(r.reason);
  };

  return (
    <div className="money-bench">
      {error && (
        <p role="alert" className="display money-bench__alert text-danger">
          {error}
        </p>
      )}

      <div className="money-col">
        <Module
          label="Coming in"
          aria-label="Coming in"
          hint="What customers paid this month."
          className="money-fill"
        >
          <Stat
            label="This month"
            value={income}
            delta={last ? income - total(last.income) : undefined}
            deltaLabel="against last month"
            format={formatMoney}
          />
          <Display scroll label="Income by line" className="money-ledger">
            <p className="money-ledger__head">By plan</p>
            <Bars
              bare
              rows={PLAN_INCOME.map((c) => ({
                label: INCOME_NAMES[c],
                value: now.income[c],
                color: c === "business" ? "var(--brand-2)" : undefined,
                testId: `income-${c}`,
                onOpen: (el) => open({ kind: "income", category: c }, el),
              }))}
            />
            {others.length > 0 && (
              <>
                <p className="money-ledger__head">Other income</p>
                <Bars
                  bare
                  rows={others.map((c) => ({
                    label: INCOME_NAMES[c],
                    value: now.income[c],
                    testId: `income-${c}`,
                    onOpen: (el) => open({ kind: "income", category: c }, el),
                  }))}
                />
              </>
            )}
            <p className="money-ledger__head">By customer group</p>
            <Bars
              bare
              rows={GROUP_IDS.map((g) => ({
                label: GROUPS[g].many,
                value: now.revenueByGroup[g],
                onOpen: (el) => open({ kind: "group", group: g }, el),
              }))}
            />
            <p className="money-ledger__head">By region</p>
            <Bars
              bare
              rows={REGION_IDS.map((r) => ({
                label: REGION_NAMES[r],
                value: now.revenueByRegion[r],
                color: "var(--brand-2)",
                onOpen: (el) => open({ kind: "region", region: r }, el),
              }))}
            />
          </Display>
        </Module>
      </div>

      <div className="money-col">
        <Module
          label="Going out"
          aria-label="Going out"
          hint="What running the company cost this month."
          className="money-fill"
        >
          <Stat
            label="This month"
            value={costs}
            delta={last ? costs - total(last.costs) : undefined}
            deltaLabel="against last month"
            format={formatMoney}
            downIsGood
          />
          <Display scroll label="Costs by line" className="money-ledger">
            <Bars
              bare
              rows={COST_CATEGORIES.filter(
                (c) => !OPTIONAL_COSTS.has(c) || now.costs[c] > 0 || (last?.costs[c] ?? 0) > 0,
              ).map((c) => ({
                label: COST_NAMES[c],
                value: now.costs[c],
                color: "var(--danger)",
                testId: `cost-${c}`,
                onOpen: (el) => open({ kind: "cost", category: c }, el),
              }))}
            />
          </Display>
          <p className="money-note">
            Serving costs grow with your customers, free ones too, and with the size of the AI
            behind their plan.
          </p>
        </Module>
        {financed && (
          <Module
            label="Loans and investors"
            aria-label="Loans and investors"
            weight="thin"
            hint="Money that is neither income nor a cost: rescues in, repayments out."
          >
            <Bars
              rows={[
                { label: "Loans", value: now.financing.loans, color: "var(--good)" },
                { label: "Investors", value: now.financing.investors, color: "var(--good)" },
                { label: "Repaid", value: now.financing.repaid, color: "var(--danger)" },
              ].filter((r) => r.value > 0)}
            />
            {company.loan && (
              <p className="money-note">
                {`Owed: ${formatMoney(company.loan.balance)} · repaying ${formatMoney(company.loan.principalPerMonth)} a month, plus ${Math.round(company.loan.monthlyRate * 100)}% interest`}
              </p>
            )}
            {ownership < 1 && (
              <p className="money-note">{`Investors own ${Math.round((1 - ownership) * 100)}% of the company.`}</p>
            )}
          </Module>
        )}
        <Module
          label="Report cards"
          aria-label="Report cards"
          hint="Each month after your first launch gets a grade from profit, growth and how happy customers are."
          className="money-cards"
        >
          {company.reports.length === 0 ? (
            <p className="money-note">
              Your first report card comes at the end of the month after you launch.
            </p>
          ) : (
            <Display scroll label="Report cards, newest first" className="money-rack">
              <ul>
                {[...company.reports].reverse().map((r) => (
                  <li key={r.month}>
                    <StampCard
                      grade={r.grade}
                      caption={monthName(r.month, true)}
                      color={gradeColor(r.grade)}
                      label={`${monthName(r.month)}: ${r.grade}`}
                      onClick={(e) => open({ kind: "report", month: r.month }, e.currentTarget)}
                    />
                  </li>
                ))}
              </ul>
            </Display>
          )}
        </Module>
      </div>

      <div className="money-col">
        <Module
          label="Profit by month"
          actions={
            <PushButton
              variant="secondary"
              onClick={(e) => open({ kind: "money" }, e.currentTarget)}
            >
              Cash day by day
            </PushButton>
          }
        >
          <Display className="money-months">
            <MonthBars
              label="Profit by month"
              bars={[...company.ledger.months, now].map((m) => ({
                month: m.month,
                value: profitOf(m),
              }))}
              format={formatMoney}
              onOpen={(month, el) => open({ kind: "month", month }, el)}
              height={84}
            />
          </Display>
          <p className="money-note">Money in, less out. Each month opens its report.</p>
        </Module>
        <Module
          label="Computers"
          aria-label="Computers"
          hint="Where your AIs run: your data centers first, then rented cloud."
        >
          <Computers view={view} />
        </Module>
        <Module
          label="Your AIs on sale"
          hint="Up to three at once."
          className="money-fill"
          actions={
            <PushButton
              onClick={() => setLaunching(true)}
              disabled={blocked !== null}
              title={blocked ?? undefined}
            >
              Launch an AI
            </PushButton>
          }
        >
          {company.products.length === 0 ? (
            <p className="money-note">Nothing on sale yet. Launch an AI to get customers.</p>
          ) : (
            <Display scroll label="Products" className="money-products">
              <ul>
                {company.products.map((p) => {
                  const v = company.versions.find((x) => x.id === p.versionId);
                  const state =
                    p.status === "live"
                      ? "On sale"
                      : p.status === "countdown"
                        ? `Launches in ${p.launchDay - view.world.day} game days`
                        : "Retired";
                  return (
                    <li
                      key={p.id}
                      className={`product-plate ${p.status === "retired" ? "product-plate--retired" : ""}`}
                    >
                      <Lamp
                        state={
                          p.status === "live" ? "ready" : p.status === "countdown" ? "wait" : "off"
                        }
                        label={state}
                      />
                      <button
                        type="button"
                        aria-label={`Open ${p.name}`}
                        onClick={(e) => open({ kind: "product", productId: p.id }, e.currentTarget)}
                        className="product-plate__open"
                      >
                        <span className="product-plate__name player-text">{p.name}</span>
                        <span className="product-plate__meta">{`${v?.label ?? ""} · ${state}`}</span>
                      </button>
                      {p.status !== "retired" && (
                        <PushButton
                          variant="quiet"
                          aria-label={`Retire ${p.name}`}
                          title="Take it off sale. Move its plans to another AI first."
                          onClick={() => void send({ type: "retireProduct", productId: p.id })}
                        >
                          Retire
                        </PushButton>
                      )}
                    </li>
                  );
                })}
              </ul>
            </Display>
          )}
        </Module>
      </div>

      <div className="money-col">
        <Module
          label="Plans and prices"
          hint="Which AI each plan sells, and what it costs customers."
        >
          <div className="plan-rows">
            {PLAN_IDS.map((plan) => (
              <PlanRow key={plan} view={view} plan={plan} send={send} />
            ))}
          </div>
        </Module>
        <FundPanel view={view} />
      </div>

      {launching && <LaunchDialog view={view} onClose={() => setLaunching(false)} />}
    </div>
  );
}

/** One plan: the AI it sells on keys (or Closed), and its price on a knob within its bounds. */
function PlanRow({ view, plan, send }: { view: WorldView; plan: PlanId; send: Send }) {
  const company = view.me.company;
  const words = planWords(view, plan);
  const live = (company?.products ?? []).filter((p) => p.status === "live");
  const current = company?.plans[plan].productId ?? null;
  const paid = plan === "free" ? null : plan;
  const bounds = paid ? getBalance(view.world.balance).economy.prices[paid] : null;
  const saved = paid ? (company?.plans[paid].price ?? bounds?.min ?? 0) : 0;
  const [price, setPrice] = useState(saved);

  return (
    <div className="plan-row">
      <p className="engrave">{words.name}</p>
      <KeyRow
        legend={`AI for the ${words.name.toLowerCase()} plan`}
        className="keyrow--names"
        options={[
          { label: "Closed", value: "" },
          ...live.map((p) => ({ label: p.name, value: p.id })),
        ]}
        value={current ?? ""}
        onChange={(id) => void send({ type: "setPlan", plan, productId: id || null })}
      />
      <p className="plan-row__what">{words.what}</p>
      {paid && bounds && (
        <div className="plan-row__price">
          <Knob
            label={`${words.name} price`}
            value={price}
            min={bounds.min}
            max={bounds.max}
            onChange={setPrice}
            format={formatMoney}
            size={40}
            showLabel={false}
          />
          <span className="plan-row__per">
            {`a month${paid === "business" ? " per seat" : ""} · ${formatMoney(bounds.min)}–${formatMoney(bounds.max)}`}
          </span>
          <PushButton
            variant={price === saved ? "secondary" : "primary"}
            aria-label={`Set the ${paid} price`}
            onClick={() => void send({ type: "setPrice", plan: paid, price })}
          >
            Set
          </PushButton>
        </div>
      )}
    </div>
  );
}

/** What the plans earn; other income (apps, the fund) is listed apart once there is some. */
const PLAN_INCOME = ["personal", "business"] as const;
const used = (c: string, _as: "plan") => (PLAN_INCOME as readonly string[]).includes(c);
/** Money lines that only appear once the company spends on them. */
const OPTIONAL_COSTS = new Set<string>([
  "events",
  "sponsorships",
  "looks",
  "appCosts",
  "investmentLosses",
  "team",
  "licenses",
]);
