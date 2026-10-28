import { type GroupId, PLAN_IDS } from "@airace/protocol";
import { Bars } from "../components/Bars.tsx";
import { DayChart } from "../components/charts/DayChart.tsx";
import { Face } from "../components/Face.tsx";
import { PostCard } from "../components/PostCard.tsx";
import { Stat } from "../components/Stat.tsx";
import { formatCount } from "../format.ts";
import { GROUPS, groupColors } from "../groups.ts";
import { daysOf } from "../history.ts";
import {
  aboutPeople,
  caresAbout,
  groupCustomers,
  groupSatisfaction,
  groupWilling,
  myPosts,
} from "../people.ts";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, Section, WindowFrame, type WindowProps } from "./WindowFrame.tsx";

const PLAN_NAMES = { free: "Free", personal: "Personal", business: "Business" } as const;

/** One customer group: how many are yours, how they feel, who they are and what they say. */
export function GroupWindow({ entry, view, group: g }: WindowProps & { group: GroupId }) {
  const open = useWindows((s) => s.open);
  const company = view.me.company;
  const pub = view.companies.find((c) => c.id === company?.id);
  if (!company || !pub) return <GoneWindow entry={entry} />;
  const days = daysOf(company);
  const color = groupColors(pub.palette)[g];
  const customers = groupCustomers(view, g);
  const willing = groupWilling(view, g);
  const sat = groupSatisfaction(view, g);
  const person = view.spotlights.find((p) => p.group === g);
  const posts = myPosts(view, (p) => p.author.group === g);
  return (
    <WindowFrame entry={entry} title={GROUPS[g].many} subtitle={GROUPS[g].wants} wide>
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat
          label="Your customers"
          value={customers}
          format={formatCount}
          testId="group-customers"
        />
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted">
            Of those willing
          </p>
          <p className="font-display text-3xl font-bold">
            {willing > 0 ? `${((100 * customers) / willing).toFixed(2)}%` : "0%"}
          </p>
          <p className="text-sm text-muted">of {aboutPeople(willing)} who would use an AI</p>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted">Mood</p>
          {sat === null ? (
            <p className="mt-2 text-sm text-muted">No AI of yours is on sale yet.</p>
          ) : (
            <p className="mt-1 flex items-center gap-2">
              <Face sat={sat} size={36} />
              <span className="font-display text-3xl font-bold">{Math.round(sat * 100)}%</span>
            </p>
          )}
        </div>
      </div>

      <div className="mt-6">
        <DayChart
          label={`${GROUPS[g].many} by day`}
          days={days.map((d) => d.day)}
          format={formatCount}
          series={[
            {
              id: g,
              label: GROUPS[g].many,
              color,
              values: days.map((d) => d.customers[g]),
            },
          ]}
        />
      </div>

      <div className="grid gap-x-6 sm:grid-cols-2">
        <Section title="On each plan">
          <Bars
            format={formatCount}
            rows={PLAN_IDS.map((p) => ({
              label: PLAN_NAMES[p],
              value: company.customers.byGroup[g][p],
              color,
            }))}
          />
        </Section>
        <Section title="What they care about">
          <ul className="flex flex-wrap gap-2">
            {caresAbout(view, g).map((c) => (
              <li key={c} className="chip">
                {c}
              </li>
            ))}
          </ul>
        </Section>
      </div>

      {person && (
        <Section title="Someone in this group">
          <div
            data-testid="spotlight"
            className="flex items-center gap-3 rounded-xl border border-border bg-surface-2/60 p-3"
          >
            <span
              aria-hidden="true"
              className="grid h-11 w-11 shrink-0 place-items-center rounded-full font-display text-lg font-bold text-on-brand"
              style={{ background: color }}
            >
              {person.name.slice(0, 1)}
            </span>
            <div className="min-w-0 text-sm">
              <p>
                <span className="player-text font-semibold">{person.name}</span>{" "}
                <span className="player-text text-xs text-muted">{person.handle}</span>
              </p>
              <p className="player-text text-xs text-muted">{person.bio}</p>
            </div>
          </div>
        </Section>
      )}

      <Section title="What they're saying">
        {posts.length === 0 ? (
          <p className="text-sm text-muted">Nothing yet from this group about your AIs.</p>
        ) : (
          <ul className="space-y-3">
            {posts.map((p) => (
              <li key={p.id}>
                <PostCard post={p} onOpen={(el) => open({ kind: "post", postId: p.id }, el)} />
              </li>
            ))}
          </ul>
        )}
      </Section>
    </WindowFrame>
  );
}
