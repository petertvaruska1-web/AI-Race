import { motion } from "motion/react";
import type { Tab } from "../route.ts";

/** The game's stations, in the order of the station knob. */
export const SCREENS: { id: Tab; label: string }[] = [
  { id: "company", label: "Company" },
  { id: "market", label: "Market" },
  { id: "money", label: "Money" },
  { id: "lab", label: "AI Lab" },
  { id: "play", label: "Playground" },
  { id: "bench", label: "Benchmarks" },
  { id: "rivals", label: "Rivals" },
  { id: "arena", label: "Arena" },
  { id: "directory", label: "AI Directory" },
];

/** Where the station knob points for the station at `index` (30 degrees a station). */
export function stationAngle(index: number): number {
  return -120 + index * 30;
}

/**
 * The station knob (design §7.1): a large knob that turns to the open station, over the nine
 * stations engraved with their lamps. The links stay real links, so the keyboard, the address
 * bar and the tests all work.
 */
export function SideNav({ worldId, tab }: { worldId: string; tab: Tab }) {
  const index = Math.max(
    0,
    SCREENS.findIndex((s) => s.id === tab),
  );
  return (
    <nav aria-label="Screens" className="station-nav">
      <div className="bench-knob station-knob" aria-hidden="true">
        <div
          className="bench-knob__cap"
          style={{ transform: `rotate(${stationAngle(index)}deg)` }}
        />
      </div>
      <ul className="station-links">
        {SCREENS.map((s) => {
          const open = tab === s.id;
          return (
            <li key={s.id}>
              <a
                href={`#/w/${worldId}/${s.id}`}
                aria-current={open ? "page" : undefined}
                className={`station-link ${open ? "station-link--open" : ""}`}
              >
                {open && (
                  <motion.span
                    data-pill
                    layoutId="screen-pill"
                    className="station-link__pill"
                    transition={{ type: "spring", stiffness: 500, damping: 38 }}
                  />
                )}
                <span className="lamp" data-state={open ? "on" : "off"} aria-hidden="true" />
                <span className="relative">{s.label}</span>
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
