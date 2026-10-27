import { useEffect } from "react";
import { useRoute } from "./route.ts";
import { useSession } from "./store/session.ts";
import { useWorld } from "./store/world.ts";
import { applyCompanyTheme } from "./theme/palette.ts";

/** Routes between screens; later tasks fill in each one. */
export function App() {
  const route = useRoute();
  const me = useSession((s) => s.me);
  const view = useWorld((s) => s.view);

  // C5: once the player has a company, every screen takes its colors.
  const palette =
    route.screen === "world"
      ? (view?.companies.find((c) => c.id === view.me.company?.id)?.palette ?? null)
      : null;
  useEffect(() => applyCompanyTheme(palette), [palette]);

  return (
    <main className="mx-auto max-w-5xl p-6">
      <h1 className="text-3xl font-bold">AI Race</h1>
      <p className="text-muted">{me ? `Signed in as ${me.displayName}` : "Loading…"}</p>
    </main>
  );
}
