import type { NewsView, WorldView } from "@airace/protocol";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { useWindows } from "../store/windows.ts";

const SHOW_MS = 7000;
const MOST = 3;

/** Milestones worth a burst: a launch, the first paying customer, customer marks, good viral news. */
function celebrate(n: NewsView): boolean {
  return n.topic !== "viral" || n.sentiment > 0;
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
 * News about the player's own company pops up in the corner as it happens
 * (immersion design); milestones burst. A toast opens the customers window.
 */
export function Toasts({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const mine = view.me.company?.id;
  const seen = useRef<Set<string> | null>(null);
  const [toasts, setToasts] = useState<NewsView[]>([]);

  useEffect(() => {
    const news = view.feed.filter((f): f is NewsView => f.kind === "news");
    if (seen.current === null) {
      // What was already there when the world opened is old news.
      seen.current = new Set(news.map((n) => n.id));
      return;
    }
    const known = seen.current;
    const fresh = news.filter((n) => !known.has(n.id) && n.companyId === mine);
    for (const n of news) known.add(n.id);
    if (fresh.length > 0) setToasts((t) => [...t, ...fresh].slice(-MOST));
  }, [view.feed, mine]);

  useEffect(() => {
    if (toasts.length === 0) return;
    const t = setTimeout(() => setToasts((all) => all.slice(1)), SHOW_MS);
    return () => clearTimeout(t);
  }, [toasts]);

  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-30 flex w-80 max-w-[calc(100vw-2rem)] flex-col gap-2">
      <AnimatePresence initial={false}>
        {toasts.map((n) => (
          <motion.div
            key={n.id}
            role="status"
            layout
            initial={{ opacity: 0, x: 60, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 60 }}
            transition={{ type: "spring", stiffness: 380, damping: 30 }}
            className="window pointer-events-auto relative"
          >
            {celebrate(n) && <Burst color={n.sentiment < 0 ? "var(--danger)" : "var(--good)"} />}
            <button
              type="button"
              onClick={(e) => {
                setToasts((all) => all.filter((x) => x.id !== n.id));
                open({ kind: "customers" }, e.currentTarget);
              }}
              className="flex w-full items-start gap-3 px-4 py-3 text-left"
            >
              <span
                aria-hidden="true"
                className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: n.sentiment < 0 ? "var(--danger)" : "var(--good)" }}
              />
              <span>
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-muted">
                  News
                </span>
                <span className="player-text block font-display font-semibold">{n.headline}</span>
              </span>
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
