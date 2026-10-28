import { GROUP_IDS, PLAN_IDS, REGION_IDS, REGION_NAMES, type WorldView } from "@airace/protocol";
import { AnimatePresence, motion } from "motion/react";
import { Button } from "../components/Button.tsx";
import { Face } from "../components/Face.tsx";
import { Panel } from "../components/Panel.tsx";
import { RegionMap, type RegionMarks } from "../components/RegionMap.tsx";
import { Stat } from "../components/Stat.tsx";
import { formatCount, formatMoney } from "../format.ts";
import { GROUPS, groupColors } from "../groups.ts";
import { aboutPeople, caresAbout, groupCustomers, groupSatisfaction } from "../people.ts";
import { regionCustomers, shareByRegion } from "../regions.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";

const PLAN_NAMES = { free: "Free", personal: "Personal", business: "Business" } as const;

/** The most dots a group's crowd draws. */
const CROWD_MOST = 120;

/** People per dot (1, 2 or 5 times a power of ten), so the biggest crowd stays within `CROWD_MOST` dots. */
export function crowdStep(most: number): number {
  for (let mag = 1; ; mag *= 10) {
    for (const k of [1, 2, 5]) if (most / (k * mag) <= CROWD_MOST) return k * mag;
  }
}

/** A group's customers as a crowd of dots in a sunflower spiral; new ones pop in. */
function Crowd({ people, step, color }: { people: number; step: number; color: string }) {
  const dots = people > 0 ? Math.max(1, Math.round(people / step)) : 0;
  const w = 220;
  const h = 96;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="mt-3 h-24 w-full" aria-hidden="true">
      {Array.from({ length: dots }, (_, i) => {
        const r = 4 * Math.sqrt(i + 0.5);
        const a = i * 2.399963;
        return (
          <circle
            // biome-ignore lint/suspicious/noArrayIndexKey: dots are alike; each keeps its place in the spiral
            key={i}
            data-dot
            className="dot"
            cx={w / 2 + r * Math.cos(a) * 1.9}
            cy={h / 2 + r * Math.sin(a) * 0.95}
            r={3.2}
            fill={color}
            style={{ animationDelay: `${(i % 16) * 35}ms, ${(i * 211) % 3400}ms` }}
          />
        );
      })}
      {dots === 0 && (
        <text
          x={w / 2}
          y={h / 2 + 4}
          textAnchor="middle"
          className="fill-[var(--muted)] text-[11px]"
        >
          No customers here yet
        </text>
      )}
    </svg>
  );
}

/**
 * The market (game-layer design §5, national in Phase 1): who could buy, who
 * does, and why. Each group's customers show as a crowd; a card opens its group.
 */
export function Market({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const company = view.me.company;
  const pub = view.companies.find((c) => c.id === company?.id);
  if (!company || !pub) return null;
  const { adoptable, adoption } = view.world.market;
  const yours = company.customers.total;
  const history = [...company.ledger.months.map((m) => m.customersEnd), yours];
  const colors = groupColors(pub.palette);
  const step = crowdStep(Math.max(0, ...GROUP_IDS.map((g) => groupCustomers(view, g))));
  return (
    <div className="space-y-5">
      <Panel title="Across the country" aria-label="Across the country">
        <div className="grid gap-5 sm:grid-cols-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-muted">
              People willing to use an AI
            </p>
            <p className="font-display text-3xl font-bold">{aboutPeople(adoptable)}</p>
            <p className="text-sm text-muted">
              {Math.round(adoption * 100)}% of everyone, and growing
            </p>
          </div>
          <div className="lift relative -m-2 rounded-xl p-2">
            <Stat label="Your customers" value={yours} format={formatCount} />
            <Growth values={history} />
            <button
              type="button"
              aria-label="Open customers"
              className="absolute inset-0 rounded-xl"
              onClick={(e) => open({ kind: "customers" }, e.currentTarget)}
            />
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-muted">Your share</p>
            <p className="font-display text-3xl font-bold">
              {adoptable > 0 ? `${((100 * yours) / adoptable).toFixed(2)}%` : "0%"}
            </p>
            <p className="text-sm text-muted">of everyone willing</p>
          </div>
        </div>
        <div className="mt-5">
          <RegionMap
            label="Your customers by region"
            values={shareByRegion(view)}
            color={pub.palette.primary}
            marks={regionMarks(view)}
            onOpen={(region, el) => open({ kind: "region", region }, el)}
            readout={(r) => {
              const n = regionCustomers(view, r);
              const people = view.world.market.regions[r].adoptable;
              return (
                <p className="text-muted">
                  <span className="tabular font-semibold text-text">{formatCount(n)}</span>{" "}
                  customers · {people > 0 ? `${((100 * n) / people).toFixed(2)}%` : "0%"} of those
                  willing
                </p>
              );
            }}
          />
        </div>
        <p className="mt-3 text-xs text-muted">
          <span className="font-semibold text-text">
            {`Each dot is ${step === 1 ? "1 person" : `${formatCount(step)} people`}`}
          </span>{" "}
          in the crowds below. Brighter regions are where more of the people willing use your AI.
        </p>
      </Panel>

      <Campaigns view={view} />

      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        {GROUP_IDS.map((g) => {
          const person = view.spotlights.find((p) => p.group === g);
          const sat = groupSatisfaction(view, g);
          const byPlan = company.customers.byGroup[g];
          return (
            <Panel
              key={g}
              title={GROUPS[g].many}
              aria-label={GROUPS[g].many}
              className="lift relative"
            >
              <p className="text-xs text-muted">Cares about: {caresAbout(view, g).join(", ")}</p>
              <div className="mt-4 flex items-end justify-between gap-3">
                <div>
                  <p className="font-display text-2xl font-bold">
                    {formatCount(groupCustomers(view, g))}
                  </p>
                  <p className="text-xs text-muted">
                    {PLAN_IDS.map((p) => `${PLAN_NAMES[p]} ${formatCount(byPlan[p])}`).join(" · ")}
                  </p>
                </div>
                {sat !== null && (
                  <div className="flex items-center gap-2">
                    <Face sat={sat} />
                    <span className="font-semibold">{Math.round(sat * 100)}%</span>
                  </div>
                )}
              </div>
              <Crowd people={groupCustomers(view, g)} step={step} color={colors[g]} />
              {person && (
                <div className="mt-3 rounded-xl border border-border bg-surface-2/60 p-3 text-sm">
                  <p>
                    <span className="player-text font-semibold">{person.name}</span>{" "}
                    <span className="player-text text-xs text-muted">{person.handle}</span>
                  </p>
                  <p className="player-text text-xs text-muted">{person.bio}</p>
                </div>
              )}
              <button
                type="button"
                aria-label={`Open ${GROUPS[g].many}`}
                className="absolute inset-0 rounded-[inherit]"
                onClick={(e) => open({ kind: "group", group: g }, e.currentTarget)}
              />
            </Panel>
          );
        })}
      </div>
    </div>
  );
}

/** The running ad campaigns, with days left and spend, and a way to start one. */
function Campaigns({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const socket = useSocket();
  const campaigns = view.me.company?.campaigns ?? [];
  return (
    <Panel
      title="Ad campaigns"
      aria-label="Ad campaigns"
      hint="Ads make more people aware of you in a region or group. Up to five at once."
      actions={
        <Button onClick={(e) => open({ kind: "campaign" }, e.currentTarget)}>
          New ad campaign
        </Button>
      }
    >
      {campaigns.length === 0 ? (
        <p className="text-sm text-muted">No campaigns running. Word of mouth does the work.</p>
      ) : (
        <ul className="space-y-3">
          <AnimatePresence initial={false}>
            {campaigns.map((k) => {
              const where = k.region === "all" ? "Everywhere" : REGION_NAMES[k.region];
              const who = k.group === "all" ? "everyone" : GROUPS[k.group].many.toLowerCase();
              const left = Math.max(0, k.endDay - view.world.day);
              const done = (view.world.day - k.startDay) / Math.max(1, k.endDay - k.startDay);
              return (
                <motion.li
                  key={k.id}
                  layout
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: 30 }}
                  className="rounded-xl border border-border bg-surface-2/50 p-3"
                >
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
                    <span className="font-semibold">{`${where} · ${who}`}</span>
                    <span className="text-muted">{`${formatMoney(k.budgetPerDay)} a day`}</span>
                    <span className="text-muted">{`${formatMoney(k.spent)} spent`}</span>
                    <span className="ml-auto text-muted">{`${left} ${left === 1 ? "day" : "days"} left`}</span>
                    <Button
                      variant="ghost"
                      aria-label={`Stop the ${k.region === "all" ? "nationwide" : REGION_NAMES[k.region]} campaign`}
                      onClick={() =>
                        void socket.command(view.world.id, {
                          type: "stopCampaign",
                          campaignId: k.id,
                        })
                      }
                    >
                      Stop
                    </Button>
                  </div>
                  <div className="mt-2 h-1.5 rounded-full bg-surface">
                    <div
                      className="h-1.5 rounded-full bg-brand transition-[width] duration-700"
                      style={{ width: `${Math.min(100, 100 * done)}%` }}
                    />
                  </div>
                </motion.li>
              );
            })}
          </AnimatePresence>
        </ul>
      )}
    </Panel>
  );
}

/** Campaigns and data centers shown on the map. */
export function regionMarks(view: WorldView): RegionMarks {
  const marks: RegionMarks = {};
  const company = view.me.company;
  for (const k of company?.campaigns ?? []) {
    const regions = k.region === "all" ? REGION_IDS : [k.region];
    for (const r of regions) marks[r] = { ...marks[r], campaign: true };
  }
  for (const d of company?.dataCenters ?? []) {
    const now = marks[d.region]?.dataCenter;
    marks[d.region] = {
      ...marks[d.region],
      dataCenter: d.open || now === "open" ? "open" : "building",
    };
  }
  return marks;
}

/** Customers at each month's end, then today, as a small line drawn in when shown. */
function Growth({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const w = 160;
  const h = 32;
  const top = Math.max(1, ...values);
  const points = values
    .map((v, i) => `${(i / (values.length - 1)) * w},${h - (v / top) * (h - 2) - 1}`)
    .join(" ");
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="mt-2 h-8 w-40"
      role="img"
      aria-label="Customers by month"
    >
      <polyline
        points={points}
        fill="none"
        stroke="var(--brand)"
        strokeWidth={2}
        pathLength={1}
        className="chart-draw"
      />
    </svg>
  );
}
