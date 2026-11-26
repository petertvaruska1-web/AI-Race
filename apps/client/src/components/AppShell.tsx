import type { WorldView } from "@airace/protocol";
import { motion } from "motion/react";
import { type ReactNode, useEffect } from "react";
import { Lamp } from "../bench/Lamp.tsx";
import { reducedMotion } from "../motion.ts";
import type { Tab } from "../route.ts";
import { useWindows } from "../store/windows.ts";
import { WindowHost } from "../windows/WindowHost.tsx";
import { Button } from "./Button.tsx";
import { SideNav } from "./SideNav.tsx";
import { Toasts } from "./Toasts.tsx";
import { LighterGraphics, TopBar } from "./TopBar.tsx";
import { WorldChat } from "./WorldChat.tsx";

/**
 * Warnings that need the player sit on hazard plates under the strip: a rescue offer counting
 * down, and servers that leave questions unanswered. Each opens what deals with it.
 */
function Hazards({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const offer = view.me.company?.rescue;
  const served = view.me.company?.compute.served ?? 1;
  if (!offer && served >= 1) return null;
  const left = offer ? Math.max(0, offer.deadlineDay - view.world.day) : 0;
  return (
    <div className="station-hazards">
      {offer && (
        <div className="rescue-banner station-hazard module module--thin">
          <Lamp state="problem" label="Rescue needed" blink />
          <span className="font-semibold">{`Your cash ran out. Choose a rescue within ${left} game days.`}</span>
          <Button className="ml-auto" onClick={(e) => open({ kind: "rescue" }, e.currentTarget)}>
            Choose a rescue
          </Button>
        </div>
      )}
      {served < 1 && (
        <div className="station-hazard module module--thin">
          <Lamp state="problem" label="Servers overloaded" blink />
          <button
            type="button"
            aria-label={`Servers overloaded · ${Math.round(served * 100)}% answered`}
            onClick={(e) => open({ kind: "computers" }, e.currentTarget)}
            className="overload-chip push push--secondary text-danger"
          >
            {`Servers overloaded · ${Math.round(served * 100)}% answered`}
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * A world's station frame (design §7.1): the top strip, the station knob, and the bench of the
 * open station, filling the window. The page never scrolls; a bench that holds more than the
 * window scrolls inside itself. A new bench slides in when the station changes.
 */
export function AppShell({
  view,
  tab,
  children,
}: {
  view: WorldView;
  tab: Tab;
  children: ReactNode;
}) {
  const closeAll = useWindows((s) => s.closeAll);
  // Windows belong to the screen they were opened on.
  // biome-ignore lint/correctness/useExhaustiveDependencies: closes them whenever the screen changes
  useEffect(() => {
    closeAll();
    return closeAll;
  }, [tab, closeAll]);
  const still = reducedMotion();
  return (
    <div data-frame="station" className="station-frame">
      <TopBar view={view} />
      <Hazards view={view} />
      <aside className="station-column module">
        <SideNav worldId={view.world.id} tab={tab} />
        <div className="station-column__foot">
          <LighterGraphics />
        </div>
      </aside>
      <main className="station-bench">
        <motion.div
          key={tab}
          className="station-bench__inner"
          initial={still ? false : { x: "3%", opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.28, ease: [0.2, 0.8, 0.2, 1] }}
        >
          {children}
        </motion.div>
      </main>
      <Toasts view={view} />
      <WorldChat view={view} />
      <WindowHost view={view} />
    </div>
  );
}
