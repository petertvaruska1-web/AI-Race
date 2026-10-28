import { GROUP_IDS, type GroupId, PLAN_IDS } from "@airace/protocol";
import { Bars } from "../components/Bars.tsx";
import { Face } from "../components/Face.tsx";
import { PostCard } from "../components/PostCard.tsx";
import { dayLabel, formatCount, formatMoney } from "../format.ts";
import { GROUPS, groupColors } from "../groups.ts";
import { Orb } from "../orb/Orb.tsx";
import { myPosts } from "../people.ts";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, Section, WindowFrame, type WindowProps } from "./WindowFrame.tsx";

/** One product: its status, plans, the AI behind it, its customers and what they say. */
export function ProductWindow({ entry, view, productId }: WindowProps & { productId: string }) {
  const open = useWindows((s) => s.open);
  const company = view.me.company;
  const pub = view.companies.find((c) => c.id === company?.id);
  const product = company?.products.find((p) => p.id === productId);
  if (!company || !pub || !product) return <GoneWindow entry={entry} />;
  const version = company.versions.find((v) => v.id === product.versionId);
  const plans = PLAN_IDS.filter((plan) => company.plans[plan].productId === product.id);
  const colors = groupColors(pub.palette);
  const sat: Partial<Record<GroupId, number>> = company.satisfaction[product.id] ?? {};
  const left = product.launchDay - view.world.day;
  const posts = myPosts(view, (p) => p.productName === product.name);
  return (
    <WindowFrame
      entry={entry}
      title={product.name}
      subtitle={
        <>
          <span>
            {product.status === "live"
              ? "On sale"
              : product.status === "countdown"
                ? `Launches in ${left} ${left === 1 ? "day" : "days"}`
                : "Retired"}
          </span>
          {product.status === "live" && <span> · since {dayLabel(product.launchDay)}</span>}
        </>
      }
      wide
    >
      <div className="grid gap-5 sm:grid-cols-2">
        {version && (
          <button
            type="button"
            aria-label={`About ${version.label}`}
            onClick={(e) => open({ kind: "ai", versionId: version.id }, e.currentTarget)}
            className="row-btn items-center gap-4 p-3"
          >
            <Orb
              size={88}
              label={`Your AI ${version.label}`}
              input={{
                overall: version.overall,
                scores: version.scores,
                traits: version.traits,
                trainedTokens: version.trainedTokens,
                palette: pub.palette,
                training: null,
              }}
            />
            <span className="text-left">
              <span className="block text-xs text-muted">The AI behind it</span>
              <span className="block font-display text-xl font-bold text-brand">
                {version.label}
              </span>
              <span className="block text-xs text-muted">
                {version.overall === null
                  ? "Being scored…"
                  : `Overall score ${Math.round(version.overall)}`}
              </span>
            </span>
          </button>
        )}
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted">Plans</p>
          {plans.length === 0 ? (
            <p className="mt-1 text-sm text-muted">No plan sells it right now.</p>
          ) : (
            <ul className="mt-1 space-y-1 text-sm">
              {plans.map((plan) => (
                <li key={plan} className="flex items-baseline gap-2">
                  {plan === "free" ? (
                    <span className="font-semibold">Free</span>
                  ) : (
                    <>
                      <span className="font-semibold">
                        {`${plan === "personal" ? "Personal" : "Business"} · ${formatMoney(company.plans[plan].price)}`}
                      </span>
                      <span className="text-xs text-muted">
                        a month{plan === "business" ? " per seat" : ""}
                      </span>
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="grid gap-x-6 sm:grid-cols-2">
        <Section title="Its customers">
          <Bars
            format={formatCount}
            rows={GROUP_IDS.map((g) => ({
              label: GROUPS[g].many,
              value: plans.reduce((n, plan) => n + company.customers.byGroup[g][plan], 0),
              color: colors[g],
              onOpen: (el) => open({ kind: "group", group: g }, el),
            }))}
          />
        </Section>
        <Section title="How happy they are">
          {Object.keys(sat).length === 0 ? (
            <p className="text-sm text-muted">Customers haven't told you yet.</p>
          ) : (
            <ul className="grid grid-cols-2 gap-2 text-sm">
              {GROUP_IDS.filter((g) => sat[g] !== undefined).map((g) => (
                <li key={g} className="flex items-center gap-2">
                  <Face sat={sat[g] ?? 0} size={22} />
                  <span className="text-muted">{GROUPS[g].many}</span>
                  <span className="tabular ml-auto font-semibold">
                    {Math.round((sat[g] ?? 0) * 100)}%
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Section>
      </div>

      <Section title="What people say about it">
        {posts.length === 0 ? (
          <p className="text-sm text-muted">Nothing yet.</p>
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
