import { GROUP_IDS, type GroupId, type WorldView } from "@airace/protocol";
import { useState } from "react";
import { Button } from "../components/Button.tsx";
import { Feed } from "../components/Feed.tsx";
import { Panel } from "../components/Panel.tsx";
import { Stat } from "../components/Stat.tsx";
import { monthProfit } from "../components/TopBar.tsx";
import { formatCount, formatMoney } from "../format.ts";
import { GROUPS } from "../groups.ts";
import { Orb } from "../orb/Orb.tsx";
import { go } from "../route.ts";
import { mix } from "../theme/palette.ts";

const total = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);

/** Why the newest AI can't be launched yet, or null when it can. */
export function launchBlocked(view: WorldView): string | null {
  const latest = view.me.company?.versions.at(-1);
  if (!latest) return "Train your first AI before launching it.";
  if (latest.scores === null) return "Wait until your AI has been scored before launching it.";
  return null;
}

/** The company's home: its AI as an orb, customers and money on the left, the feed on the right. */
export function Company({ view }: { view: WorldView }) {
  const [, setLaunching] = useState(false);
  const company = view.me.company;
  const pub = view.companies.find((c) => c.id === company?.id);
  if (!company || !pub) return null;
  const latest = company.versions.at(-1);
  const run = company.runs.find((r) => r.status === "training" || r.status === "finalizing");
  const blocked = launchBlocked(view);
  const to = (tab: "lab" | "play") => go({ screen: "world", worldId: view.world.id, tab });

  const startOfMonth = company.ledger.months.at(-1)?.customersEnd ?? 0;
  const byGroup = GROUP_IDS.map((g) => [g, total(company.customers.byGroup[g])] as const);
  const profits = [...company.ledger.months, company.ledger.current].map(
    (m) => total(m.income) - total(m.costs),
  );

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)_minmax(0,1fr)]">
      <div className="space-y-5 lg:order-1">
        <Panel title="Customers">
          <Stat
            label="People using your AI"
            value={company.customers.total}
            delta={company.customers.total - startOfMonth}
            deltaLabel="this month"
            format={formatCount}
          />
          <GroupBar groups={byGroup} palette={pub.palette} />
        </Panel>
        <Panel title="Money">
          <div className="space-y-4">
            <Stat label="Cash" value={company.cash} format={formatMoney} />
            <Stat label="Profit this month" value={monthProfit(view)} format={formatMoney} />
          </div>
          <Sparkline values={profits} />
        </Panel>
      </div>

      <Panel className="flex flex-col items-center gap-4 text-center lg:order-0 xl:order-1">
        <Orb
          size={280}
          label={
            latest ? `Your AI ${latest.label}` : "A spark: your first AI is waiting to be trained"
          }
          input={{
            overall: latest?.overall ?? null,
            scores: latest?.scores ?? null,
            traits: latest?.traits ?? null,
            trainedTokens: latest?.trainedTokens ?? 0,
            palette: pub.palette,
            training: null,
          }}
        />
        {latest ? (
          <div>
            <p className="font-display text-4xl font-bold text-brand">{latest.label}</p>
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
          <p className="text-sm">
            <span className="font-semibold text-brand">Training</span> ·{" "}
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
        <Lineup view={view} />
      </Panel>

      <Panel title="What people are saying" className="lg:order-2 lg:col-span-2 xl:col-span-1">
        <div className="max-h-[70vh] overflow-y-auto pr-1">
          <Feed items={view.feed} />
        </div>
      </Panel>
    </div>
  );
}

/** One bar split by group, in shades of the company's colors, with a legend. */
function GroupBar({
  groups,
  palette,
}: {
  groups: readonly (readonly [GroupId, number])[];
  palette: { primary: string; secondary: string };
}) {
  const sum = groups.reduce((a, [, n]) => a + n, 0);
  const color = (i: number) => mix(palette.secondary, palette.primary, i / (groups.length - 1));
  return (
    <div className="mt-4">
      <div className="flex h-3 overflow-hidden rounded-full bg-surface-2" aria-hidden="true">
        {sum > 0 &&
          groups.map(([g, n], i) => (
            <div key={g} style={{ width: `${(100 * n) / sum}%`, background: color(i) }} />
          ))}
      </div>
      <ul className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        {groups.map(([g, n], i) => (
          <li key={g} className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full" style={{ background: color(i) }} />
            <span className="text-muted">{GROUPS[g].many}</span>
            <span className="tabular ml-auto">{formatCount(n)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Profit month by month, as a small line over a zero line. */
function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) {
    return (
      <p className="mt-4 text-xs text-muted">Your profit line starts after your first month.</p>
    );
  }
  const w = 240;
  const h = 56;
  const lo = Math.min(0, ...values);
  const hi = Math.max(0, ...values);
  const y = (v: number) => (hi === lo ? h / 2 : h - ((v - lo) / (hi - lo)) * (h - 4) - 2);
  const x = (i: number) => (i / (values.length - 1)) * w;
  const points = values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="mt-4 h-14 w-full"
      role="img"
      aria-label="Profit by month"
      preserveAspectRatio="none"
    >
      <line x1={0} x2={w} y1={y(0)} y2={y(0)} stroke="var(--border)" strokeDasharray="4 4" />
      <polyline points={points} fill="none" stroke="var(--brand)" strokeWidth={2} />
    </svg>
  );
}

/** The products on sale or counting down. */
function Lineup({ view }: { view: WorldView }) {
  const products = (view.me.company?.products ?? []).filter((p) => p.status !== "retired");
  if (products.length === 0) return null;
  return (
    <ul className="w-full space-y-1 border-t border-border pt-3 text-sm">
      {products.map((p) => (
        <li key={p.id} className="flex justify-between gap-3">
          <span className="player-text font-semibold">{p.name}</span>
          <span className="text-muted">
            {p.status === "live"
              ? "On sale"
              : `Launches in ${p.launchDay - view.world.day} ${p.launchDay - view.world.day === 1 ? "day" : "days"}`}
          </span>
        </li>
      ))}
    </ul>
  );
}
