import { GROUP_IDS, type GroupId, PLAN_IDS, type WorldView } from "@airace/protocol";
import { getBalance } from "@airace/sim";
import { Panel } from "../components/Panel.tsx";
import { Stat } from "../components/Stat.tsx";
import { formatCount } from "../format.ts";
import { GROUPS } from "../groups.ts";
import { CATEGORIES } from "./Benchmarks.tsx";

/** A large count of people as words ("about 33.5 million"). */
export function aboutPeople(n: number): string {
  if (n >= 1e6) return `about ${(n / 1e6).toFixed(1)} million`;
  if (n >= 1e3) return `about ${Math.round(n / 1e3)} thousand`;
  return formatCount(n);
}

const PLAN_NAMES = { free: "Free", personal: "Personal", business: "Business" } as const;

/** What a group weighs, in plain words, from the economy's own numbers. */
function caresAbout(view: WorldView, g: GroupId): string[] {
  const traits = getBalance(view.world.balance).economy.groups[g];
  const skills = Object.entries(traits.weights)
    .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
    .slice(0, 3)
    .map(([id]) => CATEGORIES.find((c) => c.id === id)?.name ?? id);
  const price =
    traits.priceSensitivity >= 1.2
      ? "a low price"
      : traits.priceSensitivity >= 0.8
        ? "a fair price"
        : "quality over price";
  const speed =
    traits.speedPreference > 0.2
      ? "fast answers"
      : traits.speedPreference < -0.2
        ? "smarts over speed"
        : null;
  return [...skills, price, ...(speed ? [speed] : [])];
}

/** The group's satisfaction, weighted by its customers on each plan (null with no AI on sale). */
function groupSatisfaction(view: WorldView, g: GroupId): number | null {
  const company = view.me.company;
  if (!company) return null;
  let weighted = 0;
  let people = 0;
  const known: number[] = [];
  for (const plan of PLAN_IDS) {
    const product = company.plans[plan].productId;
    const sat = product ? company.satisfaction[product]?.[g] : undefined;
    if (sat === undefined) continue;
    const n = company.customers.byGroup[g][plan];
    known.push(sat);
    weighted += n * sat;
    people += n;
  }
  if (known.length === 0) return null;
  return people > 0 ? weighted / people : known.reduce((a, b) => a + b, 0) / known.length;
}

function Face({ sat }: { sat: number }) {
  const mood = sat >= 0.65 ? "Happy" : sat <= 0.35 ? "Unhappy" : "Neutral";
  const mouth =
    mood === "Happy"
      ? "M8 14.5c1.2 1.4 2.5 2 4 2s2.8-.6 4-2"
      : mood === "Unhappy"
        ? "M8 16.5c1.2-1.4 2.5-2 4-2s2.8.6 4 2"
        : "M8.5 15.5h7";
  const color =
    mood === "Happy" ? "var(--good)" : mood === "Unhappy" ? "var(--danger)" : "var(--muted)";
  return (
    <svg viewBox="0 0 24 24" width={28} height={28} role="img" aria-label={mood} style={{ color }}>
      <circle cx="12" cy="12" r="9.5" fill="none" stroke="currentColor" strokeWidth={1.6} />
      <circle cx="9" cy="10" r="1.1" fill="currentColor" />
      <circle cx="15" cy="10" r="1.1" fill="currentColor" />
      <path d={mouth} fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" />
    </svg>
  );
}

/** The market (game-layer design §5, national in Phase 1): who could buy, who does, and why. */
export function Market({ view }: { view: WorldView }) {
  const company = view.me.company;
  if (!company) return null;
  const { adoptable, adoption } = view.world.market;
  const yours = company.customers.total;
  const history = [...company.ledger.months.map((m) => m.customersEnd), yours];
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
          <Stat label="Your customers" value={yours} format={formatCount} />
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-muted">Your share</p>
            <p className="font-display text-3xl font-bold">
              {adoptable > 0 ? `${((100 * yours) / adoptable).toFixed(2)}%` : "0%"}
            </p>
            <Growth values={history} />
          </div>
        </div>
        <p className="mt-4 text-xs text-muted">
          Regions arrive soon: you'll see your customers spread across the country.
        </p>
      </Panel>

      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        {GROUP_IDS.map((g) => {
          const person = view.spotlights.find((p) => p.group === g);
          const sat = groupSatisfaction(view, g);
          const byPlan = company.customers.byGroup[g];
          return (
            <Panel key={g} title={GROUPS[g].many} aria-label={GROUPS[g].many}>
              <p className="text-xs text-muted">Cares about: {caresAbout(view, g).join(", ")}</p>
              <div className="mt-4 flex items-end justify-between gap-3">
                <div>
                  <p className="font-display text-2xl font-bold">
                    {formatCount(PLAN_IDS.reduce((a, p) => a + byPlan[p], 0))}
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
              {person && (
                <div className="mt-4 rounded-xl border border-border bg-surface-2/60 p-3 text-sm">
                  <p>
                    <span className="player-text font-semibold">{person.name}</span>{" "}
                    <span className="player-text text-xs text-muted">{person.handle}</span>
                  </p>
                  <p className="player-text text-xs text-muted">{person.bio}</p>
                </div>
              )}
            </Panel>
          );
        })}
      </div>
    </div>
  );
}

/** Customers at each month's end, then today, as a small line. */
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
      <polyline points={points} fill="none" stroke="var(--brand)" strokeWidth={2} />
    </svg>
  );
}
