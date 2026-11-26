import { GROUP_IDS, type GroupId, type Palette, type WorldView } from "@airace/protocol";
import { useState } from "react";
import { Display } from "../bench/Display.tsx";
import { KeyRow } from "../bench/KeyRow.tsx";
import { Lamp } from "../bench/Lamp.tsx";
import { Module } from "../bench/Module.tsx";
import { Readout } from "../bench/Readout.tsx";
import { Trace } from "../bench/Trace.tsx";
import { Button } from "../components/Button.tsx";
import { Feed } from "../components/Feed.tsx";
import { FloatDelta } from "../components/FloatDelta.tsx";
import { Logo } from "../components/Logo.tsx";
import { Panel } from "../components/Panel.tsx";
import { RegionMap } from "../components/RegionMap.tsx";
import { Stat } from "../components/Stat.tsx";
import { monthProfit } from "../components/TopBar.tsx";
import { Core } from "../core/Core.tsx";
import { formatCount, formatMoney, monthName } from "../format.ts";
import { GROUPS, groupColors } from "../groups.ts";
import { daysOf } from "../history.ts";
import { profitOf, total } from "../money.ts";
import { arrivalsFrom } from "../orb/params.ts";
import { regionCustomers, shareByRegion } from "../regions.ts";
import { go } from "../route.ts";
import { LicensesPanel } from "../spending/Licensing.tsx";
import { AppsPanel, OrbFrame } from "../spending/Spending.tsx";
import { TeamPanel } from "../spending/Team.tsx";
import { useWindows } from "../store/windows.ts";
import { GoalProgress } from "../surprises/Goals.tsx";
import { TrophyShelf } from "../surprises/Trophies.tsx";
import { useReadable } from "../theme/glass.ts";
import { gradeColor } from "../windows/ReportWindow.tsx";
import { LaunchDialog, launchBlocked } from "./LaunchDialog.tsx";
import { regionMarks } from "./Market.tsx";

/**
 * The Company station (design §10): the AI core in its cradle at the center, customers and money
 * on the left, what people are saying on the right, the team and the rest in a drawer below.
 * Everything opens its window; new customers stream into the core.
 */
export function Company({ view }: { view: WorldView }) {
  const [launching, setLaunching] = useState(false);
  const open = useWindows((s) => s.open);
  const company = view.me.company;
  const pub = view.companies.find((c) => c.id === company?.id);
  if (!company || !pub) return null;
  const latest = company.versions.at(-1);
  const run = company.runs.find((r) => r.status === "training" || r.status === "finalizing");
  const blocked = launchBlocked(view);
  const to = (tab: "lab" | "play") => go({ screen: "world", worldId: view.world.id, tab });

  const startOfMonth = company.ledger.months.at(-1)?.customersEnd ?? 0;
  const byGroup = GROUP_IDS.map((g) => [g, total(company.customers.byGroup[g])] as const);
  const profits = [...company.ledger.months, company.ledger.current].map(profitOf);
  const lastCard = company.reports.at(-1);
  const orb = (
    <Core
      size={300}
      label={latest ? `Your AI ${latest.label}` : "A spark: your first AI is waiting to be trained"}
      input={{
        overall: latest?.overall ?? null,
        scores: latest?.scores ?? null,
        traits: latest?.traits ?? null,
        trainedTokens: latest?.trainedTokens ?? 0,
        palette: pub.palette,
        training: null,
        arrivals: arrivalsFrom(daysOf(company), pub.palette),
      }}
    />
  );

  return (
    <div className="grid gap-3.5 xl:grid-cols-[300px_minmax(0,1fr)_340px]">
      <div className="flex min-w-0 flex-col gap-3.5">
        <Panel title="Customers">
          <div className="lift relative -m-1.5 rounded-lg p-1.5">
            <Stat
              label="People using your AI"
              value={company.customers.total}
              delta={company.customers.total - startOfMonth}
              deltaLabel="this month"
              format={formatCount}
              testId="customers"
            />
            <FloatDelta value={company.customers.total} format={formatCount} />
            <button
              type="button"
              aria-label="Open customers"
              className="absolute inset-0 rounded-lg"
              onClick={(e) => open({ kind: "customers" }, e.currentTarget)}
            />
          </div>
          <GroupBar groups={byGroup} palette={pub.palette} />
          <Display className="mt-3 px-2 py-2">
            <RegionMap
              small
              label="Customers by region"
              values={shareByRegion(view)}
              color={pub.palette.primary}
              marks={regionMarks(view)}
              onOpen={(region, el) => open({ kind: "region", region }, el)}
              readout={(r) => (
                <p className="text-muted">
                  <span className="tabular font-semibold text-text">
                    {formatCount(regionCustomers(view, r))}
                  </span>{" "}
                  customers
                </p>
              )}
            />
          </Display>
        </Panel>
        <Panel title="Money">
          <div className="lift relative -m-1.5 rounded-lg p-1.5">
            <div className="grid grid-cols-2 gap-3">
              <Readout label="Cash" value={formatMoney(company.cash)} size="sm" />
              <Readout
                label="Profit this month"
                value={formatMoney(monthProfit(view))}
                size="sm"
                tone={monthProfit(view) < 0 ? "danger" : monthProfit(view) > 0 ? "good" : "lamp"}
              />
            </div>
            {profits.length < 2 ? (
              <p className="mt-3 text-xs text-muted">
                Your profit line starts after your first month.
              </p>
            ) : (
              <Display className="mt-3 px-2 py-1.5">
                <Trace values={profits} label="Profit by month" zeroLine />
              </Display>
            )}
            <button
              type="button"
              aria-label="Open money"
              className="absolute inset-0 rounded-lg"
              onClick={(e) => open({ kind: "money" }, e.currentTarget)}
            />
          </div>
          {pub.ownership < 1 && (
            <p className="chip mt-3 inline-block">{`You own ${Math.round(pub.ownership * 100)}%`}</p>
          )}
          {lastCard && (
            <button
              type="button"
              aria-label={`Last report card: ${lastCard.grade}`}
              onClick={(e) => open({ kind: "report", month: lastCard.month }, e.currentTarget)}
              className="row-btn mt-3"
            >
              <span
                className="grid h-9 w-9 place-items-center rounded-md border-2 font-dot text-lg font-extrabold"
                style={{
                  color: gradeColor(lastCard.grade),
                  borderColor: gradeColor(lastCard.grade),
                }}
              >
                {lastCard.grade}
              </span>
              <span>
                <span className="block font-semibold">Report card</span>
                <span className="block text-xs text-muted">{monthName(lastCard.month)}</span>
              </span>
              <span aria-hidden="true" className="ml-auto text-muted">
                ›
              </span>
            </button>
          )}
        </Panel>
      </div>

      <Module weight="heavy" className="flex min-w-0 flex-col items-center gap-3 text-center">
        <div className="core-cradle">
          <OrbFrame look={company.looks.orb} color={pub.palette.primary}>
            {latest ? (
              <button
                type="button"
                aria-label={`About your AI ${latest.label}`}
                title="Open your AI"
                onClick={(e) => open({ kind: "ai", versionId: latest.id }, e.currentTarget)}
                className="orb-button rounded-full"
              >
                {orb}
              </button>
            ) : (
              orb
            )}
          </OrbFrame>
        </div>
        {latest ? (
          <div className="flex flex-col items-center gap-2">
            <Display className="px-5 py-2">
              <p className="dotface dotface--lg m-0">{latest.label}</p>
            </Display>
            <p className="text-sm text-muted">
              {latest.overall === null
                ? "Being scored…"
                : `Overall score ${Math.round(latest.overall)}`}
              {latest.personalityWords.length > 0 && ` · ${latest.personalityWords.join(", ")}`}
            </p>
          </div>
        ) : (
          <p className="max-w-xs text-sm text-muted">
            This spark is where your AI will be. Plan its reading and train it in the AI Lab.
          </p>
        )}
        {run && (
          <p className="flex items-center gap-2 text-sm">
            <Lamp state="on" label="Training" blink />
            <span className="font-semibold text-accent">Training</span> ·{" "}
            {Math.round(run.progress * 100)}% of its {run.endDay - run.startDay} game days
          </p>
        )}
        <div className="flex flex-wrap justify-center gap-3">
          <Button onClick={() => to("lab")}>{latest ? "Train" : "Train your first AI"}</Button>
          <Button variant="secondary" onClick={() => to("play")} disabled={!latest}>
            Chat
          </Button>
          <Button
            variant="secondary"
            disabled={blocked !== null}
            title={blocked ?? undefined}
            onClick={() => setLaunching(true)}
          >
            Launch
          </Button>
        </div>
        {blocked && latest && <p className="text-xs text-muted">{blocked}</p>}
        <Button variant="ghost" onClick={(e) => open({ kind: "customize" }, e.currentTarget)}>
          ✨ Customize your look
        </Button>
        <Lineup view={view} />
        <ArchStrip view={view} />
      </Module>

      <div className="flex min-w-0 flex-col gap-3.5">
        <Panel title="What people are saying" className="flex min-h-0 flex-col">
          <Display scroll label="What people are saying" className="max-h-[56vh] p-2">
            <Feed items={view.feed} companyId={company.id} companies={view.companies} />
          </Display>
        </Panel>
        <GoalProgress view={view} />
        <TrophyShelf view={view} />
      </div>

      <DrawerBay view={view} />
      {launching && <LaunchDialog view={view} onClose={() => setLaunching(false)} />}
    </div>
  );
}

type Drawer = "team" | "licenses" | "apps";

const DRAWERS: { label: string; value: Drawer }[] = [
  { label: "Team", value: "team" },
  { label: "Licenses", value: "licenses" },
  { label: "Partner apps", value: "apps" },
];

/**
 * The company's smaller panels in one drawer (design A2): one shown at a time, Team first. The
 * goal and the trophies stay in view beside the feed: a player should always see them.
 */
function DrawerBay({ view }: { view: WorldView }) {
  const [shown, setShown] = useState<Drawer>("team");
  return (
    <div className="min-w-0 xl:col-span-3">
      <KeyRow
        legend="Company drawer"
        options={DRAWERS}
        value={shown}
        onChange={setShown}
        className="mb-2.5"
      />
      {shown === "team" && <TeamPanel view={view} />}
      {shown === "licenses" && <LicensesPanel view={view} />}
      {shown === "apps" && <AppsPanel view={view} />}
    </div>
  );
}

/** One bar split by group, in shades of the company's colors; each group opens its window. */
function GroupBar({
  groups,
  palette,
}: {
  groups: readonly (readonly [GroupId, number])[];
  palette: Palette;
}) {
  const open = useWindows((s) => s.open);
  const sum = groups.reduce((a, [, n]) => a + n, 0);
  const colors = groupColors(palette);
  return (
    <div className="mt-4">
      <div className="flex h-3 overflow-hidden rounded-full bg-surface-2" aria-hidden="true">
        {sum > 0 &&
          groups.map(([g, n]) => (
            <div
              key={g}
              className="transition-[width] duration-700"
              style={{ width: `${(100 * n) / sum}%`, background: colors[g] }}
            />
          ))}
      </div>
      <ul className="mt-3 grid grid-cols-2 gap-x-2 gap-y-0.5 text-xs">
        {groups.map(([g, n]) => (
          <li key={g}>
            <button
              type="button"
              aria-label={`Open ${GROUPS[g].many}`}
              onClick={(e) => open({ kind: "group", group: g }, e.currentTarget)}
              className="lift-row flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left"
            >
              <span className="h-2 w-2 rounded-full" style={{ background: colors[g] }} />
              <span className="text-muted">{GROUPS[g].many}</span>
              <span className="tabular ml-auto">{formatCount(n)}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** The products on sale or counting down; each opens its window. */
function Lineup({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const products = (view.me.company?.products ?? []).filter((p) => p.status !== "retired");
  if (products.length === 0) return null;
  return (
    <ul className="w-full space-y-1.5 border-t border-border pt-3 text-sm">
      {products.map((p) => {
        const left = p.launchDay - view.world.day;
        return (
          <li key={p.id}>
            <button
              type="button"
              aria-label={`Open ${p.name}`}
              onClick={(e) => open({ kind: "product", productId: p.id }, e.currentTarget)}
              className="row-btn justify-between"
            >
              <span className="player-text font-semibold">{p.name}</span>
              <span className="text-muted">
                {p.status === "live"
                  ? "On sale"
                  : `Launches in ${left} ${left === 1 ? "day" : "days"}`}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/** The arch-rival against you, in one line; it opens the Rivals screen. */
function ArchStrip({ view }: { view: WorldView }) {
  const readable = useReadable();
  const arch = view.companies.find((c) => c.rival?.archetype);
  const me = view.companies.find((c) => c.id === view.me.company?.id);
  if (!arch || !me) return null;
  return (
    <button
      type="button"
      onClick={() => go({ screen: "world", worldId: view.world.id, tab: "rivals" })}
      className="lift mt-4 flex w-full items-center gap-3 rounded-xl border border-border px-3 py-2 text-left text-sm"
    >
      <Logo logo={arch.logo} palette={arch.palette} size={28} />
      <span className="min-w-0 flex-1">
        <span className="block text-xs text-muted">Your arch-rival</span>
        <span className="font-semibold" style={{ color: readable(arch.palette.primary) }}>
          {arch.name}
        </span>
      </span>
      <span className="tabular text-right text-xs">
        {formatCount(arch.customers)} customers
        <span className="block text-muted">you: {formatCount(me.customers)}</span>
      </span>
    </button>
  );
}
