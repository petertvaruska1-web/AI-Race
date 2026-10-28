import {
  COST_CATEGORIES,
  type CostCategory,
  GROUP_IDS,
  INCOME_CATEGORIES,
  type IncomeCategory,
  type MonthLedgerView,
} from "@airace/protocol";
import { Bars } from "../components/Bars.tsx";
import { DayChart } from "../components/charts/DayChart.tsx";
import { MonthBars } from "../components/charts/MonthBars.tsx";
import { Stat } from "../components/Stat.tsx";
import { formatCompact, formatCount, formatMoney, monthName, signed } from "../format.ts";
import { GROUPS } from "../groups.ts";
import { daysOf } from "../history.ts";
import { COST_NAMES, INCOME_NAMES, MONEY_WHAT, profitOf, total } from "../money.ts";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, Section, WindowFrame, type WindowProps } from "./WindowFrame.tsx";

const moneyAxis = (n: number) => `${n < 0 ? "-" : ""}$${formatCompact(Math.abs(n))}`;

function allMonths(view: WindowProps["view"]): MonthLedgerView[] {
  const ledger = view.me.company?.ledger;
  return ledger ? [...ledger.months, ledger.current] : [];
}

/** The company's money: cash day by day, profit month by month, this month's ins and outs. */
export function MoneyWindow({ entry, view }: WindowProps) {
  const open = useWindows((s) => s.open);
  const company = view.me.company;
  const pub = view.companies.find((c) => c.id === company?.id);
  if (!company || !pub) return <GoneWindow entry={entry} />;
  const days = daysOf(company);
  const now = company.ledger.current;
  const yesterday = days.length > 1 ? (days.at(-2)?.cash ?? company.cash) : company.cash;
  return (
    <WindowFrame
      entry={entry}
      title="Money"
      subtitle="Your cash day by day, and profit month by month."
      wide
    >
      <div className="grid grid-cols-3 gap-4">
        <Stat label="Cash" value={company.cash} format={formatMoney} />
        <Stat
          label="Since yesterday"
          value={company.cash - yesterday}
          format={signed(formatMoney)}
        />
        <Stat label="Profit this month" value={profitOf(now)} format={formatMoney} />
      </div>
      <div className="mt-6">
        <DayChart
          label="Cash by day"
          days={days.map((d) => d.day)}
          format={formatMoney}
          axis={moneyAxis}
          series={[
            {
              id: "cash",
              label: "Cash",
              color: pub.palette.primary,
              values: days.map((d) => d.cash),
            },
          ]}
        />
      </div>
      <Section title="Profit by month">
        <MonthBars
          label="Profit by month"
          bars={allMonths(view).map((m) => ({ month: m.month, value: profitOf(m) }))}
          format={formatMoney}
          onOpen={(month, el) => open({ kind: "month", month }, el)}
        />
      </Section>
      <div className="grid gap-x-6 sm:grid-cols-2">
        <Section title="Coming in this month">
          <Bars
            rows={INCOME_CATEGORIES.map((c) => ({
              label: INCOME_NAMES[c],
              value: now.income[c],
              color: "var(--good)",
              onOpen: (el) => open({ kind: "income", category: c }, el),
            }))}
          />
        </Section>
        <Section title="Going out this month">
          <Bars
            rows={COST_CATEGORIES.map((c) => ({
              label: COST_NAMES[c],
              value: now.costs[c],
              color: "var(--danger)",
              onOpen: (el) => open({ kind: "cost", category: c }, el),
            }))}
          />
        </Section>
      </div>
    </WindowFrame>
  );
}

/** One month's report: profit, what came in and from whom, what went out, customers at the close. */
export function MonthWindow({ entry, view, month }: WindowProps & { month: string }) {
  const open = useWindows((s) => s.open);
  const company = view.me.company;
  const m = allMonths(view).find((x) => x.month === month);
  if (!company || !m) return <GoneWindow entry={entry} />;
  const current = m.month === company.ledger.current.month;
  return (
    <WindowFrame
      entry={entry}
      title={monthName(m.month)}
      subtitle={current ? "This month so far." : "A closed month."}
      wide
    >
      <div className="grid grid-cols-3 gap-4">
        <Stat label="Profit" value={profitOf(m)} format={formatMoney} testId="month-profit" />
        <Stat label="Came in" value={total(m.income)} format={formatMoney} />
        <Stat label="Went out" value={total(m.costs)} format={formatMoney} downIsGood />
      </div>
      <div className="grid gap-x-6 sm:grid-cols-2">
        <Section title="Came in, by plan">
          <Bars
            rows={INCOME_CATEGORIES.map((c) => ({
              label: INCOME_NAMES[c],
              value: m.income[c],
              color: "var(--good)",
              testId: `in-${c}`,
              onOpen: (el) => open({ kind: "income", category: c }, el),
            }))}
          />
        </Section>
        <Section title="Went out">
          <Bars
            rows={COST_CATEGORIES.map((c) => ({
              label: COST_NAMES[c],
              value: m.costs[c],
              color: "var(--danger)",
              testId: `out-${c}`,
              onOpen: (el) => open({ kind: "cost", category: c }, el),
            }))}
          />
        </Section>
        <Section title="Came in, by customer group">
          <Bars
            rows={GROUP_IDS.map((g) => ({
              label: GROUPS[g].many,
              value: m.revenueByGroup[g],
              onOpen: (el) => open({ kind: "group", group: g }, el),
            }))}
          />
        </Section>
        <Section title="Customers at the close">
          {current ? (
            <p className="text-sm text-muted">The month is still open.</p>
          ) : (
            <p className="font-display text-3xl font-bold" data-testid="month-customers">
              {formatCount(m.customersEnd)}
            </p>
          )}
        </Section>
      </div>
    </WindowFrame>
  );
}

/** One kind of income or cost, month by month. */
export function CategoryWindow({
  entry,
  view,
  side,
  category,
}: WindowProps &
  ({ side: "income"; category: IncomeCategory } | { side: "costs"; category: CostCategory })) {
  const open = useWindows((s) => s.open);
  const months = allMonths(view);
  if (months.length === 0) return <GoneWindow entry={entry} />;
  const name =
    side === "income"
      ? INCOME_NAMES[category as IncomeCategory]
      : COST_NAMES[category as CostCategory];
  const value = (m: MonthLedgerView) =>
    side === "income" ? m.income[category as IncomeCategory] : m.costs[category as CostCategory];
  const now = months.at(-1);
  const last = months.at(-2);
  return (
    <WindowFrame entry={entry} title={name} subtitle={MONEY_WHAT[category]} wide>
      <div className="grid grid-cols-2 gap-4">
        <Stat
          label="This month"
          value={now ? value(now) : 0}
          delta={now && last ? value(now) - value(last) : undefined}
          deltaLabel="against last month"
          format={formatMoney}
          downIsGood={side === "costs"}
        />
        <Stat
          label="Over these months"
          value={months.reduce((a, m) => a + value(m), 0)}
          format={formatMoney}
        />
      </div>
      <Section title="Month by month">
        <MonthBars
          label={`${name} by month`}
          bars={months.map((m) => ({ month: m.month, value: value(m) }))}
          format={formatMoney}
          color={() => (side === "income" ? "var(--good)" : "var(--danger)")}
          onOpen={(month, el) => open({ kind: "month", month }, el)}
        />
      </Section>
    </WindowFrame>
  );
}
