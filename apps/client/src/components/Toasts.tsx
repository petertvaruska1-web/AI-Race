import type { NewsView, WorldView } from "@airace/protocol";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { monthName } from "../format.ts";
import { useWindows, type WindowSpec } from "../store/windows.ts";

const SHOW_MS = 7000;
const MOST = 3;

type Toast = {
  id: string;
  label: string;
  headline: string;
  good: boolean;
  /** Milestones and top grades burst. */
  burst: boolean;
  opens: WindowSpec;
};

/** A news story about the player's company as a toast; milestones burst (bad viral news doesn't). */
function newsToast(n: NewsView): Toast {
  return {
    id: n.id,
    label: "News",
    headline: n.headline,
    good: n.sentiment >= 0,
    burst: n.sentiment > 0,
    opens: { kind: "customers" },
  };
}

const SPARKS = Array.from({ length: 14 }, (_, i) => {
  const a = (i / 14) * Math.PI * 2;
  const d = 34 + 18 * ((i * 7) % 3);
  return { id: i, x: Math.cos(a) * d, y: Math.sin(a) * d };
});

function Burst({ color }: { color: string }) {
  return (
    <span aria-hidden="true" className="pointer-events-none absolute top-1/2 left-6">
      {SPARKS.map((s) => (
        <motion.span
          key={s.id}
          className="absolute h-1.5 w-1.5 rounded-full"
          style={{ background: s.id % 2 ? color : "var(--brand-2)" }}
          initial={{ x: 0, y: 0, opacity: 1, scale: 1 }}
          animate={{ x: s.x, y: s.y, opacity: 0, scale: 0.4 }}
          transition={{ duration: 0.9, ease: "easeOut" }}
        />
      ))}
    </span>
  );
}

/**
 * News about the player's own company, and each new report card, pop up in
 * the corner as they happen (immersion design); good ones burst. A toast
 * opens what it is about.
 */
export function Toasts({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const mine = view.me.company?.id;
  const reports = view.me.company?.reports;
  const seen = useRef<Set<string> | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    const news = view.feed.filter((f): f is NewsView => f.kind === "news");
    const cards = reports ?? [];
    const keys = [...news.map((n) => n.id), ...cards.map((r) => `report-${r.month}`)];
    if (seen.current === null) {
      // What was already there when the world opened is old news.
      seen.current = new Set(keys);
      return;
    }
    const known = seen.current;
    const fresh: Toast[] = [
      ...news.filter((n) => !known.has(n.id) && n.companyId === mine).map(newsToast),
      ...cards
        .filter((r) => !known.has(`report-${r.month}`))
        .map((r) => ({
          id: `report-${r.month}`,
          label: "Report card",
          headline: `Your ${monthName(r.month)} report card: ${r.grade}`,
          good: r.grade !== "D" && r.grade !== "F",
          burst: r.grade === "A",
          opens: { kind: "report", month: r.month } as WindowSpec,
        })),
    ];
    for (const k of keys) known.add(k);
    if (fresh.length > 0) setToasts((t) => [...t, ...fresh].slice(-MOST));
  }, [view.feed, mine, reports]);

  useEffect(() => {
    if (toasts.length === 0) return;
    const t = setTimeout(() => setToasts((all) => all.slice(1)), SHOW_MS);
    return () => clearTimeout(t);
  }, [toasts]);

  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-30 flex w-80 max-w-[calc(100vw-2rem)] flex-col gap-2">
      <AnimatePresence initial={false}>
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            role="status"
            layout
            initial={{ opacity: 0, x: 60, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 60 }}
            transition={{ type: "spring", stiffness: 380, damping: 30 }}
            className="window pointer-events-auto relative"
          >
            {t.burst && <Burst color={t.good ? "var(--good)" : "var(--danger)"} />}
            <button
              type="button"
              onClick={(e) => {
                setToasts((all) => all.filter((x) => x.id !== t.id));
                open(t.opens, e.currentTarget);
              }}
              className="flex w-full items-start gap-3 px-4 py-3 text-left"
            >
              <span
                aria-hidden="true"
                className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: t.good ? "var(--good)" : "var(--danger)" }}
              />
              <span>
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-muted">
                  {t.label}
                </span>
                <span className="player-text block font-display font-semibold">{t.headline}</span>
              </span>
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
