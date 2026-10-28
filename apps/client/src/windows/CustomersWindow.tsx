import { GROUP_IDS } from "@airace/protocol";
import { DayChart } from "../components/charts/DayChart.tsx";
import { Stat } from "../components/Stat.tsx";
import { formatCount, signed } from "../format.ts";
import { GROUPS, groupColors } from "../groups.ts";
import { type DayPoint, daysOf } from "../history.ts";
import { groupCustomers } from "../people.ts";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, Section, WindowFrame, type WindowProps } from "./WindowFrame.tsx";

const everyone = (d: DayPoint | undefined) =>
  d ? GROUP_IDS.reduce((a, g) => a + d.customers[g], 0) : 0;

/** All the player's customers, day by day and stacked by group. */
export function CustomersWindow({ entry, view }: WindowProps) {
  const open = useWindows((s) => s.open);
  const company = view.me.company;
  const pub = view.companies.find((c) => c.id === company?.id);
  if (!company || !pub) return <GoneWindow entry={entry} />;
  const days = daysOf(company);
  const colors = groupColors(pub.palette);
  const today = everyone(days.at(-1));
  const yesterday = days.length > 1 ? everyone(days.at(-2)) : today;
  const startOfMonth = company.ledger.months.at(-1)?.customersEnd ?? 0;
  return (
    <WindowFrame
      entry={entry}
      title="Customers"
      subtitle="People using your AI, day by day, by group."
      wide
    >
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Now" value={company.customers.total} format={formatCount} />
        <Stat label="Since yesterday" value={today - yesterday} format={signed(formatCount)} />
        <Stat
          label="This month"
          value={company.customers.total - startOfMonth}
          format={signed(formatCount)}
        />
        <Stat label="Paying" value={days.at(-1)?.paying ?? 0} format={formatCount} />
      </div>
      <div className="mt-6">
        <DayChart
          label="Customers by day"
          days={days.map((d) => d.day)}
          stacked
          format={formatCount}
          series={GROUP_IDS.map((g) => ({
            id: g,
            label: GROUPS[g].many,
            color: colors[g],
            values: days.map((d) => d.customers[g]),
          }))}
        />
      </div>
      <Section title="By group">
        <ul className="grid gap-2 sm:grid-cols-2">
          {GROUP_IDS.map((g) => (
            <li key={g}>
              <button
                type="button"
                aria-label={`Open ${GROUPS[g].many}`}
                onClick={(e) => open({ kind: "group", group: g }, e.currentTarget)}
                className="row-btn"
              >
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: colors[g] }} />
                <span>{GROUPS[g].many}</span>
                <span className="tabular ml-auto font-semibold">
                  {formatCount(groupCustomers(view, g))}
                </span>
                <span aria-hidden="true" className="text-muted">
                  ›
                </span>
              </button>
            </li>
          ))}
        </ul>
      </Section>
    </WindowFrame>
  );
}
