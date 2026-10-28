import type { WorldView } from "@airace/protocol";
import { motion } from "motion/react";
import { type ReactNode, useEffect, useId, useRef } from "react";
import { type OpenWindow, useWindows } from "../store/windows.ts";

export type WindowProps = { entry: OpenWindow; view: WorldView };

/**
 * A window: it grows out of what was clicked and shrinks away. Back shows the
 * window below it; the close button closes them all.
 */
export function WindowFrame({
  entry,
  title,
  subtitle,
  wide = false,
  children,
}: {
  entry: OpenWindow;
  title: ReactNode;
  subtitle?: ReactNode;
  wide?: boolean;
  children: ReactNode;
}) {
  const back = useWindows((s) => s.back);
  const closeAll = useWindows((s) => s.closeAll);
  const depth = useWindows((s) => s.stack.length);
  const ref = useRef<HTMLElement>(null);
  const titleId = useId();
  useEffect(() => {
    ref.current?.focus();
  }, []);
  const from = entry.from;
  const start = from
    ? { x: from.x - window.innerWidth / 2, y: from.y - window.innerHeight / 2, scale: 0.15 }
    : { x: 0, y: 24, scale: 0.94 };
  return (
    <motion.section
      ref={ref}
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      tabIndex={-1}
      className={`window relative max-h-[88vh] w-full overflow-y-auto p-6 outline-none ${wide ? "max-w-4xl" : "max-w-2xl"}`}
      initial={{ opacity: 0, ...start }}
      animate={{ opacity: 1, x: 0, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.94, transition: { duration: 0.12 } }}
      transition={{ type: "spring", stiffness: 360, damping: 32 }}
    >
      <header className="mb-5 flex items-start gap-3">
        {depth > 1 && (
          <button type="button" aria-label="Back" onClick={back} className="icon-btn">
            ←
          </button>
        )}
        <div className="min-w-0 flex-1">
          <h2 id={titleId} className="player-text font-display text-2xl font-bold">
            {title}
          </h2>
          {subtitle && <p className="mt-1 text-sm text-muted">{subtitle}</p>}
        </div>
        <button type="button" aria-label="Close" onClick={closeAll} className="icon-btn">
          ✕
        </button>
      </header>
      {children}
    </motion.section>
  );
}

/** For a window whose subject has gone (a post scrolled away, a product retired long ago). */
export function GoneWindow({ entry }: { entry: OpenWindow }) {
  return (
    <WindowFrame entry={entry} title="Not here any more">
      <p className="text-sm text-muted">What this window showed is no longer around.</p>
    </WindowFrame>
  );
}

/** A small heading inside a window. */
export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mt-6">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">{title}</h3>
      {children}
    </section>
  );
}
