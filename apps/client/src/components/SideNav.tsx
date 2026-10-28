import { motion } from "motion/react";
import type { ReactNode } from "react";
import type { Tab } from "../route.ts";

const stroke = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
} as const;

/** The game's screens, with small line icons. Arena and Rivals join in Phase 3. */
export const SCREENS: { id: Tab; label: string; icon: ReactNode }[] = [
  {
    id: "company",
    label: "Company",
    icon: (
      <>
        <circle cx="12" cy="12" r="4" {...stroke} />
        <circle cx="12" cy="12" r="8.5" {...stroke} opacity={0.5} />
      </>
    ),
  },
  {
    id: "market",
    label: "Market",
    icon: (
      <>
        <circle cx="9" cy="9" r="3" {...stroke} />
        <circle cx="16.5" cy="10.5" r="2.2" {...stroke} />
        <path
          d="M3.5 19c.8-3 3-4.5 5.5-4.5s4.7 1.5 5.5 4.5M14.5 15c2.6-.4 4.6.8 5.5 3.5"
          {...stroke}
        />
      </>
    ),
  },
  {
    id: "money",
    label: "Money",
    icon: <path d="M4 18l5-5 4 3 7-8M15 8h5v5" {...stroke} />,
  },
  {
    id: "lab",
    label: "AI Lab",
    icon: (
      <path
        d="M9.5 3.5h5M10.5 3.5v6L5 19a1.5 1.5 0 0 0 1.3 2h11.4a1.5 1.5 0 0 0 1.3-2l-5.5-9.5v-6"
        {...stroke}
      />
    ),
  },
  {
    id: "play",
    label: "Playground",
    icon: <path d="M5 5h14v10H10l-4 4v-4H5z" {...stroke} />,
  },
  {
    id: "bench",
    label: "Benchmarks",
    icon: <path d="M5 20V12M10 20V6M15 20v-9M20 20V4" {...stroke} />,
  },
];

/** The screens of a world; on narrow windows it lies across the top. */
export function SideNav({ worldId, tab }: { worldId: string; tab: Tab }) {
  return (
    <nav
      aria-label="Screens"
      className="flex gap-1 overflow-x-auto px-3 py-2 md:flex-col md:px-3 md:py-4"
    >
      {SCREENS.map((s) => (
        <a
          key={s.id}
          href={`#/w/${worldId}/${s.id}`}
          aria-current={tab === s.id ? "page" : undefined}
          className={`relative flex items-center gap-3 whitespace-nowrap rounded-xl px-3 py-2 text-sm transition ${
            tab === s.id
              ? "font-semibold text-text"
              : "text-muted hover:bg-surface-2 hover:text-text"
          }`}
        >
          {tab === s.id && (
            <motion.span
              data-pill
              layoutId="screen-pill"
              className="absolute inset-0 rounded-xl bg-brand-soft shadow-[var(--glow)]"
              transition={{ type: "spring", stiffness: 500, damping: 38 }}
            />
          )}
          <svg
            viewBox="0 0 24 24"
            width={20}
            height={20}
            aria-hidden="true"
            className={`relative ${tab === s.id ? "text-brand" : ""}`}
          >
            {s.icon}
          </svg>
          <span className="relative">{s.label}</span>
        </a>
      ))}
    </nav>
  );
}
