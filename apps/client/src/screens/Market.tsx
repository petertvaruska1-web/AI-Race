import {
  GROUP_IDS,
  type GroupId,
  REGION_IDS,
  REGION_NAMES,
  type WorldView,
} from "@airace/protocol";
import { AnimatePresence, motion } from "motion/react";
import { Display } from "../bench/Display.tsx";
import { Module } from "../bench/Module.tsx";
import { PushButton } from "../bench/PushButton.tsx";
import { Readout } from "../bench/Readout.tsx";
import { Trace } from "../bench/Trace.tsx";
import { Face } from "../components/Face.tsx";
import { RegionMap } from "../components/RegionMap.tsx";
import { RegionReadout } from "../components/RegionReadout.tsx";
import { formatCount, formatMoney } from "../format.ts";
import { GROUPS, groupColors } from "../groups.ts";
import { aboutPeople, groupCustomers, groupSatisfaction } from "../people.ts";
import { leaderTops, type RegionMarks } from "../regions.ts";
import { SPONSOR_WORDS, SponsorPanel } from "../spending/Spending.tsx";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { lampColor } from "../theme/glass.ts";

/** The dots of a group's crowd display: a 20 × 6 dot matrix. */
const CROWD_COLUMNS = 20;
const CROWD_MOST = 120;
/** Where each next person lights up on the matrix: scattered, the same every time. */
const CROWD_ORDER = Array.from({ length: CROWD_MOST }, (_, i) => (i * 37) % CROWD_MOST);

/** People per dot (1, 2 or 5 times a power of ten), so the biggest crowd stays within `CROWD_MOST` dots. */
export function crowdStep(most: number): number {
  for (let mag = 1; ; mag *= 10) {
    for (const k of [1, 2, 5]) if (most / (k * mag) <= CROWD_MOST) return k * mag;
  }
}

/**
 * A group's customers on a dot-matrix display (design §10 Market): one lit dot per step of people,
 * the rest of the 120 dark; a new one lights up where it lands.
 */
function CrowdDots({ people, step, color }: { people: number; step: number; color: string }) {
  const lit = people > 0 ? Math.min(CROWD_MOST, Math.max(1, Math.round(people / step))) : 0;
  const on = new Set(CROWD_ORDER.slice(0, lit));
  const rows = CROWD_MOST / CROWD_COLUMNS;
  return (
    <div className="market-crowd">
      <svg viewBox={`0 0 ${CROWD_COLUMNS * 7} ${rows * 7}`} aria-hidden="true">
        {Array.from({ length: CROWD_MOST }, (_, i) => {
          const x = (i % CROWD_COLUMNS) * 7 + 3.5;
          const y = Math.floor(i / CROWD_COLUMNS) * 7 + 3.5;
          return on.has(i) ? (
            <circle
              // biome-ignore lint/suspicious/noArrayIndexKey: each dot keeps its place on the matrix
              key={i}
              data-dot
              className="market-crowd__on"
              cx={x}
              cy={y}
              r={2.7}
              fill={color}
              style={{ color }}
            />
          ) : (
            // biome-ignore lint/suspicious/noArrayIndexKey: each dot keeps its place on the matrix
            <circle key={i} data-off className="market-crowd__off" cx={x} cy={y} r={2.2} />
          );
        })}
      </svg>
      {lit === 0 && <p className="market-crowd__empty">No customers here yet</p>}
    </div>
  );
}

/** One group of customers (a module named by the group): how many, their mood, their crowd. */
function GroupModule({
  view,
  group,
  step,
  color,
}: {
  view: WorldView;
  group: GroupId;
  step: number;
  color: string;
}) {
  const open = useWindows((s) => s.open);
  const person = view.spotlights.find((p) => p.group === group);
  const sat = groupSatisfaction(view, group);
  const people = groupCustomers(view, group);
  return (
    <Module aria-label={GROUPS[group].many} weight="thin" className="market-group">
      <div className="market-group__head">
        <span className="engrave">{GROUPS[group].many}</span>
        {sat !== null && (
          <span className="market-group__mood">
            <Face sat={sat} size={18} />
            <span>{`${Math.round(sat * 100)}%`}</span>
          </span>
        )}
      </div>
      <Display className="market-group__display">
        <span className="dotface dotface--sm">{formatCount(people)}</span>
        <CrowdDots people={people} step={step} color={color} />
      </Display>
      {person && (
        <div className="market-group__person">
          <p className="player-text m-0 truncate">
            <span className="font-semibold">{person.name}</span>{" "}
            <span className="text-muted">{person.handle}</span>
          </p>
          <p className="player-text market-group__bio">{person.bio}</p>
        </div>
      )}
      <button
        type="button"
        aria-label={`Open ${GROUPS[group].many}`}
        className="absolute inset-0 rounded-[inherit]"
        onClick={(e) => open({ kind: "group", group }, e.currentTarget)}
      />
    </Module>
  );
}

/**
 * The Market station (design §10): who could buy, who does, and where. The 3D map stands at its
 * middle; readouts for the country above it; ads and sponsorships beside it; the six groups of
 * customers, each with its crowd on a dot-matrix display, below.
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
    <div className="market-bench">
      <section aria-label="Across the country" className="market-readouts">
        <Module weight="thin" className="market-readout">
          <Readout label="People willing to use an AI" value={aboutPeople(adoptable)} size="sm" />
          <p className="market-readout__note">{`${Math.round(adoption * 100)}% of everyone, and growing`}</p>
        </Module>
        <Module weight="thin" className="market-readout lift relative">
          <Readout label="Your customers" value={formatCount(yours)} />
          {history.length > 1 && (
            <Trace values={history} label="Customers by month" className="market-readout__trace" />
          )}
          <button
            type="button"
            aria-label="Open customers"
            className="absolute inset-0 rounded-[inherit]"
            onClick={(e) => open({ kind: "customers" }, e.currentTarget)}
          />
        </Module>
        <Module weight="thin" className="market-readout">
          <Readout
            label="Your share"
            value={adoptable > 0 ? `${((100 * yours) / adoptable).toFixed(2)}%` : "0%"}
          />
          <p className="market-readout__note">of everyone willing</p>
        </Module>
      </section>

      <div className="market-middle">
        <Module weight="heavy" label="Who leads each region" className="market-map">
          <RegionMap
            label="Who leads each region"
            tops={leaderTops(view)}
            marks={regionMarks(view)}
            markColor={lampColor(pub.palette)}
            onOpen={(region, el) => open({ kind: "region", region }, el)}
            readout={(r) => <RegionReadout view={view} region={r} />}
          />
          <p className="market-map__note">
            A region's height shows how many of its willing people use its leader's AI.
          </p>
        </Module>
        <div className="market-side">
          <Campaigns view={view} />
          <SponsorPanel view={view} />
        </div>
      </div>

      <section aria-label="Your customers by group" className="market-groups">
        <p className="market-groups__note">
          {`Each dot is ${step === 1 ? "1 person" : `${formatCount(step)} people`}`}
        </p>
        <div className="market-groups__row">
          {GROUP_IDS.map((g) => (
            <GroupModule key={g} view={view} group={g} step={step} color={colors[g]} />
          ))}
        </div>
      </section>
    </div>
  );
}

/** The running ad campaigns, with days left and spend, and a way to start one. */
function Campaigns({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const socket = useSocket();
  const campaigns = view.me.company?.campaigns ?? [];
  return (
    <Module
      label="Ad campaigns"
      aria-label="Ad campaigns"
      hint="Ads make people aware of you. Up to five at once."
      className="market-side__module"
    >
      {campaigns.length === 0 ? (
        <p className="m-0 text-sm text-muted">No campaigns running. Word of mouth does the work.</p>
      ) : (
        <Display scroll label="Running ad campaigns" className="market-list">
          <ul className="m-0 list-none space-y-3 p-0">
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
                  >
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
                      <span className="font-semibold">{`${where} · ${who}`}</span>
                      <span className="ml-auto text-muted">{`${left} ${left === 1 ? "day" : "days"} left`}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-x-3 text-xs text-muted">
                      <span>{`${formatMoney(k.budgetPerDay)} a day`}</span>
                      <span>{`${formatMoney(k.spent)} spent`}</span>
                      <PushButton
                        variant="quiet"
                        className="ml-auto"
                        aria-label={`Stop the ${k.region === "all" ? "nationwide" : REGION_NAMES[k.region]} campaign`}
                        onClick={() =>
                          void socket.command(view.world.id, {
                            type: "stopCampaign",
                            campaignId: k.id,
                          })
                        }
                      >
                        Stop
                      </PushButton>
                    </div>
                    <div className="market-list__bar">
                      <div style={{ width: `${Math.min(100, 100 * done)}%` }} />
                    </div>
                  </motion.li>
                );
              })}
            </AnimatePresence>
          </ul>
        </Display>
      )}
      <PushButton
        className="market-side__action"
        onClick={(e) => open({ kind: "campaign" }, e.currentTarget)}
      >
        New ad campaign
      </PushButton>
    </Module>
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
  for (const x of company?.sponsorships ?? []) {
    if (view.world.day < x.endDay)
      marks[x.region] = { ...marks[x.region], sponsor: SPONSOR_WORDS[x.kind].icon };
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
