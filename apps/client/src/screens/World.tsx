import { useEffect } from "react";
import type { Route } from "../route.ts";
import { useSocket } from "../store/session.ts";
import { useWorld } from "../store/world.ts";
import { applyCompanyTheme } from "../theme/palette.ts";
import { Lobby } from "./Lobby.tsx";

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
    <main className="p-10">
      <h1 className="text-2xl font-bold">{view.world.name}</h1>
    </main>
  );
}
