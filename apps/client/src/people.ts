import {
  type GroupId,
  PLAN_IDS,
  type PostView,
  REGION_IDS,
  type WorldView,
} from "@airace/protocol";
import { getBalance } from "@airace/sim";
import { formatCount } from "./format.ts";
import { CATEGORIES } from "./screens/Benchmarks.tsx";

// Who the customers are, in plain words, from the view and the economy's own numbers.

/** A large count of people as words ("about 33.5 million"). */
export function aboutPeople(n: number): string {
  if (n >= 1e6) return `about ${(n / 1e6).toFixed(1)} million`;
  if (n >= 1e3) return `about ${Math.round(n / 1e3)} thousand`;
  return formatCount(n);
}

/** What a group weighs, from the economy's own numbers. */
export function caresAbout(view: WorldView, g: GroupId): string[] {
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
export function groupSatisfaction(view: WorldView, g: GroupId): number | null {
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

/** People in the group willing to use an AI today, across every region. */
export function groupWilling(view: WorldView, g: GroupId): number {
  const mix = getBalance(view.world.balance).economy.groupMix;
  return REGION_IDS.reduce(
    (n, r) => n + view.world.market.regions[r].adoptable * (mix[r][g] ?? 0),
    0,
  );
}

/** The player's customers in a group, over every plan. */
export function groupCustomers(view: WorldView, g: GroupId): number {
  const byPlan = view.me.company?.customers.byGroup[g];
  return byPlan ? PLAN_IDS.reduce((a, p) => a + byPlan[p], 0) : 0;
}

/** The newest posts about the player's own AIs that match, newest first. */
export function myPosts(view: WorldView, match: (p: PostView) => boolean, n = 3): PostView[] {
  const mine = view.me.company?.id;
  return view.feed
    .filter((f): f is PostView => f.kind === "post" && f.companyId === mine && match(f))
    .slice(-n)
    .reverse();
}
