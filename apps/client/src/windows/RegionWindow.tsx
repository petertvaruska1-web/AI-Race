import { GROUP_IDS, REGION_NAMES, REGION_PLACES, type RegionId } from "@airace/protocol";
import { getBalance } from "@airace/sim";
import { Bars } from "../components/Bars.tsx";
import { Button } from "../components/Button.tsx";
import { DayChart } from "../components/charts/DayChart.tsx";
import { Stat } from "../components/Stat.tsx";
import { dayLabel, formatCount, formatMoney } from "../format.ts";
import { GROUPS, groupColors } from "../groups.ts";
import { daysOf } from "../history.ts";
import { aboutPeople } from "../people.ts";
import { closenessWords, regionCustomers } from "../regions.ts";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, Section, WindowFrame, type WindowProps } from "./WindowFrame.tsx";

/** One region: its people, your customers there by group, its revenue, speed and plans. */
export function RegionWindow({ entry, view, region }: WindowProps & { region: RegionId }) {
  const open = useWindows((s) => s.open);
  const company = view.me.company;
  const pub = view.companies.find((c) => c.id === company?.id);
  if (!company || !pub) return <GoneWindow entry={entry} />;
  const e = getBalance(view.world.balance).economy;
  const willing = view.world.market.regions[region].adoptable;
  const customers = regionCustomers(view, region);
  const colors = groupColors(pub.palette);
  const days = daysOf(company);
  const centers = company.dataCenters.filter((d) => d.region === region);
  const campaigns = company.campaigns.filter((k) => k.region === region || k.region === "all");
  return (
    <WindowFrame
      entry={entry}
      title={REGION_NAMES[region]}
      subtitle={`${aboutPeople(e.population[region])} people · ${aboutPeople(willing)} willing to use an AI`}
      wide
    >
      <div className="grid grid-cols-3 gap-4">
        <Stat
          label="Your customers"
          value={customers}
          format={formatCount}
          testId="region-customers"
        />
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted">
            Of those willing
          </p>
          <p className="font-display text-3xl font-bold">
            {willing > 0 ? `${((100 * customers) / willing).toFixed(2)}%` : "0%"}
          </p>
        </div>
        <Stat
          label="Paid this month"
          value={company.ledger.current.revenueByRegion[region]}
          format={formatMoney}
          testId="region-revenue"
        />
      </div>
      <p className="mt-4 flex items-center gap-2 text-sm">
        <span
          aria-hidden="true"
          className={`h-2 w-2 rounded-full ${centers.some((d) => d.open) ? "bg-good" : "bg-muted"}`}
        />
        {closenessWords(view, region)}
      </p>
      <div className="mt-5">
        <DayChart
          label={`Customers in ${REGION_PLACES[region]} by day`}
          days={days.map((d) => d.day)}
          format={formatCount}
          series={[
            {
              id: region,
              label: REGION_NAMES[region],
              color: pub.palette.primary,
              values: days.map((d) => d.regions[region]),
            },
          ]}
        />
      </div>
      <div className="grid gap-x-6 sm:grid-cols-2">
        <Section title="Your customers here, by group">
          <Bars
            format={formatCount}
            rows={GROUP_IDS.map((g) => ({
              label: GROUPS[g].many,
              value: company.customers.byRegion[region][g],
              color: colors[g],
              onOpen: (el) => open({ kind: "group", group: g }, el),
            }))}
          />
        </Section>
        <Section title="Your plans here">
          {centers.length === 0 && campaigns.length === 0 ? (
            <p className="text-sm text-muted">No data centers or ad campaigns here yet.</p>
          ) : (
            <ul className="space-y-2 text-sm">
              {centers.map((d) => (
                <li key={d.id} className="chip block">
                  {d.open
                    ? `Data center, open since ${dayLabel(d.openDay)}`
                    : `Data center, opening ${dayLabel(d.openDay)}`}
                </li>
              ))}
              {campaigns.map((k) => (
                <li key={k.id} className="chip block">
                  {`Ad campaign, ${formatMoney(k.budgetPerDay)} a day until ${dayLabel(k.endDay)}`}
                </li>
              ))}
            </ul>
          )}
          <div className="mt-3">
            <Button
              variant="secondary"
              onClick={(e) => open({ kind: "campaign", region }, e.currentTarget)}
            >
              Advertise here
            </Button>
          </div>
        </Section>
      </div>
    </WindowFrame>
  );
}
