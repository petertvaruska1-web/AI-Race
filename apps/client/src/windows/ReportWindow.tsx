import { GROUP_IDS, type ReportCardView } from "@airace/protocol";
import { motion } from "motion/react";
import { Bars } from "../components/Bars.tsx";
import { Stat } from "../components/Stat.tsx";
import { formatCount, formatMoney, monthName, signed } from "../format.ts";
import { GROUPS } from "../groups.ts";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, Section, WindowFrame, type WindowProps } from "./WindowFrame.tsx";

type Moment = NonNullable<ReportCardView["best"]>;

export function gradeColor(grade: ReportCardView["grade"]): string {
  if (grade === "A" || grade === "B") return "var(--good)";
  if (grade === "C") return "var(--brand-2)";
  return "var(--danger)";
}

/** What drove the grade, in plain words (game commentary). */
function why(card: ReportCardView): string {
  const money =
    card.profit > 0
      ? card.lastProfit !== null && card.profit <= card.lastProfit
        ? "You made a profit, though less than last month"
        : "You made a profit"
      : card.lastProfit !== null && card.profit > card.lastProfit
        ? "You lost money, but less than last month"
        : "You lost money";
  const growth =
    card.customersStart > 0
      ? `customers ${card.customersEnd >= card.customersStart ? "grew" : "fell"} ${Math.abs(
          Math.round((100 * (card.customersEnd - card.customersStart)) / card.customersStart),
        )}%`
      : card.customersEnd > 0
        ? "your first customers arrived"
        : "no customers yet";
  const mood =
    card.mood === null
      ? "nobody to ask how they felt"
      : `customers were ${card.mood >= 0.6 ? "happy" : card.mood >= 0.45 ? "so-so" : "unhappy"} (${Math.round(card.mood * 100)}%)`;
  return `${money}; ${growth}; ${mood}.`;
}

/** A post kept on the card: the game's frame around the AI's reply, quoted as text (R1). */
function MomentCard({ moment, title }: { moment: Moment; title: string }) {
  const color =
    moment.tone === "praise"
      ? "var(--good)"
      : moment.tone === "roast"
        ? "var(--danger)"
        : "var(--brand-2)";
  return (
    <article
      className="rounded-xl border border-border bg-surface/60 p-4"
      style={{ borderLeft: `3px solid ${color}` }}
    >
      <p className="text-[11px] font-semibold uppercase tracking-wider" style={{ color }}>
        {title}
      </p>
      <p className="mt-1 text-sm">
        <span className="player-text font-semibold">{moment.author.name}</span>{" "}
        <span className="player-text text-xs text-muted">{moment.author.handle}</span>
        <span className="text-xs text-muted"> · {GROUPS[moment.author.group].one}</span>
      </p>
      <p className="mt-2 text-xs text-muted">
        asked <span className="player-text font-semibold text-text">{moment.productName}</span>:
      </p>
      <p className="player-text text-sm">{moment.question}</p>
      <p className="mt-2 text-xs text-muted">and it said:</p>
      <blockquote className="player-text mt-1 rounded-lg bg-surface-2 p-3 text-sm">
        {moment.reply === "" ? (
          <span className="italic text-muted">(nothing at all)</span>
        ) : (
          moment.reply
        )}
      </blockquote>
    </article>
  );
}

/** A month's report card: the grade lands like a stamp, then the money, growth and moments. */
export function ReportWindow({ entry, view, month }: WindowProps & { month: string }) {
  const open = useWindows((s) => s.open);
  const card = view.me.company?.reports.find((r) => r.month === month);
  if (!card) return <GoneWindow entry={entry} />;
  return (
    <WindowFrame
      entry={entry}
      title={`${monthName(card.month)} report card`}
      subtitle="How the month went, in numbers and moments."
      wide
    >
      <div className="grid items-center gap-6 sm:grid-cols-[auto_1fr]">
        <motion.div
          data-testid="grade"
          className="grid h-28 w-28 place-items-center rounded-2xl border-4 font-display text-6xl font-bold"
          style={{ color: gradeColor(card.grade), borderColor: gradeColor(card.grade) }}
          initial={{ scale: 2.4, rotate: -20, opacity: 0 }}
          animate={{ scale: 1, rotate: -6, opacity: 1 }}
          transition={{ type: "spring", stiffness: 260, damping: 14, delay: 0.15 }}
        >
          {card.grade}
        </motion.div>
        <div>
          <div className="grid grid-cols-3 gap-4">
            <Stat label="Came in" value={card.revenue} format={formatMoney} />
            <Stat label="Went out" value={card.costs} format={formatMoney} />
            <Stat
              label="Profit"
              value={card.profit}
              delta={card.lastProfit === null ? undefined : card.profit - card.lastProfit}
              deltaLabel="against last month"
              format={formatMoney}
            />
          </div>
          <p className="mt-4 text-sm">{why(card)}</p>
          <p className="mt-1 text-xs text-muted">
            Customers: {formatCount(card.customersStart)} → {formatCount(card.customersEnd)}
          </p>
        </div>
      </div>

      <Section title="Growth by group">
        <Bars
          format={signed(formatCount)}
          rows={GROUP_IDS.map((g) => ({
            label: GROUPS[g].many,
            value: card.growthByGroup[g],
            color: card.growthByGroup[g] >= 0 ? "var(--good)" : "var(--danger)",
            onOpen: (el) => open({ kind: "group", group: g }, el),
          }))}
        />
      </Section>

      {(card.best || card.worst) && (
        <Section title="The month's moments">
          <div className="grid gap-3 sm:grid-cols-2">
            {card.best && <MomentCard moment={card.best} title="Best moment" />}
            {card.worst && <MomentCard moment={card.worst} title="Worst moment" />}
          </div>
        </Section>
      )}
      {card.headline && (
        <Section title="The month's biggest story">
          <p className="player-text rounded-xl border border-border bg-surface/40 px-4 py-3 font-display font-semibold">
            {card.headline}
          </p>
        </Section>
      )}
    </WindowFrame>
  );
}
