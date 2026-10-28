import { useEffect } from "react";
import { AppShell } from "../components/AppShell.tsx";
import { Panel } from "../components/Panel.tsx";
import type { Route } from "../route.ts";
import { useSocket } from "../store/session.ts";
import { useWorld } from "../store/world.ts";
import { applyCompanyTheme } from "../theme/palette.ts";
import { Benchmarks } from "./Benchmarks.tsx";
import { Company } from "./Company.tsx";
import { Lobby } from "./Lobby.tsx";
import { Lab } from "./lab/LabLayout.tsx";
import { Money } from "./Money.tsx";
import { Playground } from "./Playground.tsx";

type WorldRoute = Extract<Route, { screen: "world" }>;

/** One world: watches it while open and themes every screen in the player's company colors (C5). */
export function WorldScreen({ route }: { route: WorldRoute }) {
  const socket = useSocket();
  const view = useWorld((s) => s.view);
  const status = useWorld((s) => s.status);
  const { worldId } = route;

  useEffect(() => {
    useWorld.getState().open(worldId);
    socket.watch(worldId);
    return () => socket.unwatch(worldId);
  }, [socket, worldId]);

  const palette = view?.companies.find((c) => c.id === view.me.company?.id)?.palette ?? null;
  useEffect(() => {
    applyCompanyTheme(palette);
    return () => applyCompanyTheme(null);
  }, [palette]);

  if (!view || view.world.id !== worldId) {
    return (
      <p className="p-10 text-muted">
        {status === "offline" ? "Reconnecting…" : "Loading the world…"}
      </p>
    );
  }
  if (view.world.phase === "setup" || !view.me.company) return <Lobby view={view} />;
  return (
    <AppShell view={view} tab={route.tab}>
      {route.tab === "company" && <Company view={view} />}
      {route.tab === "market" && (
        <Panel title="Market">
          <p className="text-muted">This screen opens soon.</p>
        </Panel>
      )}
      {route.tab === "money" && <Money view={view} />}
      {route.tab === "lab" && <Lab view={view} />}
      {route.tab === "play" && <Playground view={view} />}
      {route.tab === "bench" && <Benchmarks view={view} />}
    </AppShell>
  );
}
