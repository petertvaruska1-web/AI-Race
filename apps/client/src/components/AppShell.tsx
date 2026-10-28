import type { WorldView } from "@airace/protocol";
import { type ReactNode, useEffect } from "react";
import type { Tab } from "../route.ts";
import { useWindows } from "../store/windows.ts";
import { WindowHost } from "../windows/WindowHost.tsx";
import { Button } from "./Button.tsx";
import { SideNav } from "./SideNav.tsx";
import { Toasts } from "./Toasts.tsx";
import { TopBar } from "./TopBar.tsx";

/** While a rescue offer waits, a strip under the top bar counts down and opens the choice. */
function RescueBanner({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const offer = view.me.company?.rescue;
  if (!offer) return null;
  const left = Math.max(0, offer.deadlineDay - view.world.day);
  return (
    <div className="rescue-banner sticky top-[65px] z-10 flex flex-wrap items-center gap-3 border-b border-danger bg-bg/90 px-5 py-2 text-sm backdrop-blur-md">
      <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full bg-danger" />
      <span className="font-semibold">{`Your cash ran out. Choose a rescue within ${left} game days.`}</span>
      <Button className="ml-auto" onClick={(e) => open({ kind: "rescue" }, e.currentTarget)}>
        Choose a rescue
      </Button>
    </div>
  );
}

/** A world's frame: the top bar, the screens on the side and the open screen. */
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
  return (
    <div className="min-h-screen">
      <TopBar view={view} />
      <RescueBanner view={view} />
      <div className="md:flex">
        <aside className="border-b border-border md:sticky md:top-[65px] md:h-[calc(100vh-65px)] md:w-56 md:shrink-0 md:border-r md:border-b-0">
          <SideNav worldId={view.world.id} tab={tab} />
        </aside>
        <main className="grid-bg min-w-0 flex-1 px-5 py-6 md:px-8">
          <div key={tab} className="screen-in mx-auto max-w-6xl">
            {children}
          </div>
        </main>
      </div>
      <Toasts view={view} />
      <WindowHost view={view} />
    </div>
  );
}
