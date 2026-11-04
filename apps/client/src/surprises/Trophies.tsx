import { MILESTONE_NAMES, type MilestoneId, type WorldView } from "@airace/protocol";
import { motion } from "motion/react";
import { dayLabel } from "../format.ts";
import { useWindows } from "../store/windows.ts";
import { Section, WindowFrame, type WindowProps } from "../windows/WindowFrame.tsx";

export const MILESTONE_ICONS: Record<MilestoneId, string> = {
  firstSentence: "💬",
  firstFact: "📘",
  firstProgram: "💻",
  customers1k: "👥",
  customers100k: "🏙️",
  customers1m: "🌍",
  firstPaying: "💳",
  firstProfit: "📈",
  numberOne: "🥇",
  beatArch: "⚔️",
  dataCenter: "🏭",
  gradeA: "🅰️",
};

/** The trophy shelf: only milestones already earned (never a list of what is left, R5). */
export function TrophyShelf({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const earned = view.me.company?.milestones ?? [];
  if (earned.length === 0) return null;
  return (
    <button
      type="button"
      onClick={(e) => open({ kind: "milestones" }, e.currentTarget)}
      className="window w-full px-4 py-3 text-left"
      aria-label="Trophy shelf"
      data-testid="trophy-shelf"
    >
      <span className="block text-[11px] font-semibold uppercase tracking-wider text-muted">
        Trophy shelf
      </span>
      <ul className="mt-2 flex flex-wrap gap-3">
        {earned.map((m, k) => (
          <motion.li
            key={m.id}
            className="flex items-center gap-2 text-sm"
            initial={{ scale: 0.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 320, damping: 20, delay: 0.04 * k }}
          >
            <span aria-hidden="true">{MILESTONE_ICONS[m.id]}</span>
            <span className="font-semibold">{MILESTONE_NAMES[m.id]}</span>
            <span className="text-xs text-muted">{dayLabel(m.day)}</span>
          </motion.li>
        ))}
      </ul>
    </button>
  );
}

/** Every milestone earned, with its day and what it was earned with. */
export function MilestonesWindow({ entry, view }: WindowProps) {
  const earned = view.me.company?.milestones ?? [];
  return (
    <WindowFrame entry={entry} title="Trophy shelf" subtitle={`${earned.length} earned`}>
      <Section title="Earned">
        <ul className="space-y-3">
          {earned.map((m) => (
            <li key={m.id} className="flex items-center gap-3">
              <span aria-hidden="true" className="text-2xl">
                {MILESTONE_ICONS[m.id]}
              </span>
              <span>
                <span className="block font-semibold">{MILESTONE_NAMES[m.id]}</span>
                <span className="player-text block text-xs text-muted">
                  {dayLabel(m.day)}
                  {m.detail ? ` · ${m.detail}` : ""}
                </span>
              </span>
            </li>
          ))}
        </ul>
      </Section>
    </WindowFrame>
  );
}
