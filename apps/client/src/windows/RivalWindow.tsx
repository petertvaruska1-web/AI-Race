import {
  ARCHETYPE_WORDS,
  BENCH_CATEGORIES,
  REGION_IDS,
  type RegionId,
  STYLE_WORDS,
} from "@airace/protocol";
import { Bars } from "../components/Bars.tsx";
import { Logo } from "../components/Logo.tsx";
import { RegionMap } from "../components/RegionMap.tsx";
import { formatCount } from "../format.ts";
import { shareOfWilling } from "../rivals.ts";
import { tryRival } from "../screens/Rivals.tsx";
import { GoneWindow, Section, WindowFrame, type WindowProps } from "./WindowFrame.tsx";

const best = (scores: (number | null | undefined)[]) => {
  const known = scores.filter((v): v is number => typeof v === "number");
  return known.length > 0 ? Math.max(...known) : 0;
};

/** One rival's public profile: its AIs, their skills against yours, its customers and its news. */
export function RivalWindow({ entry, view, companyId }: WindowProps & { companyId: string }) {
  const rival = view.companies.find((c) => c.id === companyId);
  const me = view.companies.find((c) => c.id === view.me.company?.id);
  if (!rival?.rival || !me) return <GoneWindow entry={entry} />;
  const theirs = rival.products.filter((p) => p.status === "live" && p.version);
  const mine = me.products.filter((p) => p.status === "live" && p.version);
  const top = Math.max(1, ...REGION_IDS.map((r) => rival.customersByRegion[r]));
  const values = Object.fromEntries(
    REGION_IDS.map((r) => [r, rival.customersByRegion[r] / top]),
  ) as Record<RegionId, number>;
  const news = view.feed.filter((f) => f.kind === "news" && f.companyId === rival.id).slice(-6);
  const arch = rival.rival.archetype;
  return (
    <WindowFrame
      entry={entry}
      title={rival.name}
      subtitle={arch ? `Arch-rival · ${ARCHETYPE_WORDS[arch]}` : STYLE_WORDS[rival.rival.style]}
      wide
    >
      <div className="flex items-center gap-4">
        <Logo logo={rival.logo} palette={rival.palette} size={64} />
        <p className="text-sm text-muted">
          {formatCount(rival.customers)} customers ({shareOfWilling(view, rival).toFixed(2)}% of
          those willing) · reputation {rival.reputation} · {rival.followers.toLocaleString("en-US")}{" "}
          followers
        </p>
      </div>
      <Section title="Skills against yours">
        <Bars
          format={(v) => String(Math.round(v))}
          rows={BENCH_CATEGORIES.flatMap((c) => [
            {
              label: `${c}: ${rival.name}`,
              value: best(theirs.map((p) => p.version?.scores?.[c])),
              color: rival.palette.primary,
            },
            {
              label: `${c}: you`,
              value: best(mine.map((p) => p.version?.scores?.[c])),
              color: me.palette.primary,
            },
          ])}
        />
      </Section>
      <Section title="Where its customers are">
        <RegionMap
          label={`Where ${rival.name}'s customers are`}
          values={values}
          color={rival.palette.primary}
          small
        />
      </Section>
      <Section title="Its AIs">
        <ul className="space-y-2 text-sm">
          {theirs.map((p) => (
            <li key={p.id} className="flex items-center justify-between gap-2">
              <span>
                <span className="font-semibold">{p.name}</span>{" "}
                <span className="text-muted">
                  {p.version?.personalityWords.join(", ") || "no notable personality"}
                </span>
              </span>
              <button
                type="button"
                className="text-brand underline"
                onClick={() => tryRival(view, p.version?.id ?? "")}
              >
                Try it
              </button>
            </li>
          ))}
        </ul>
      </Section>
      {news.length > 0 && (
        <Section title="In the news">
          <ul className="space-y-1 text-sm">
            {news.map((n) => (
              <li key={n.id}>{n.kind === "news" ? n.headline : ""}</li>
            ))}
          </ul>
        </Section>
      )}
    </WindowFrame>
  );
}
