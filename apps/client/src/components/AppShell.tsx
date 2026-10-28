import type { WorldView } from "@airace/protocol";
import type { ReactNode } from "react";
import type { Tab } from "../route.ts";
import { SideNav } from "./SideNav.tsx";
import { TopBar } from "./TopBar.tsx";

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
  return (
    <div className="min-h-screen">
      <TopBar view={view} />
      <div className="md:flex">
        <aside className="border-b border-border md:sticky md:top-[65px] md:h-[calc(100vh-65px)] md:w-56 md:shrink-0 md:border-r md:border-b-0">
          <SideNav worldId={view.world.id} tab={tab} />
        </aside>
        <main className="grid-bg min-w-0 flex-1 px-5 py-6 md:px-8">
          <div className="mx-auto max-w-6xl">{children}</div>
        </main>
      </div>
    </div>
  );
}
