import { useEffect, useState } from "react";
import { api, type Me } from "./net/api.ts";
import { browserSocket } from "./net/socket.ts";
import { useRoute } from "./route.ts";
import { Home, Welcome } from "./screens/Home.tsx";
import { WorldScreen } from "./screens/World.tsx";
import { useSession } from "./store/session.ts";
import { bindWorldStore } from "./store/world.ts";

/** Signs the player in: one socket for the page, fed into the world store. */
export function signIn(me: Me): void {
  const socket = browserSocket(async () => {
    const fresh = await api.me();
    if (!fresh) throw new Error("signed out");
    return fresh.wsToken;
  });
  bindWorldStore(socket);
  useSession.getState().setMe(me);
  useSession.getState().setSocket(socket);
  void socket.connect();
}

export function App() {
  const route = useRoute();
  const me = useSession((s) => s.me);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    api
      .me()
      .then((existing) => {
        if (existing && !useSession.getState().me) signIn(existing);
      })
      .finally(() => setChecked(true));
  }, []);

  if (!checked) return <p className="p-10 text-muted">Loading…</p>;
  if (!me) return <Welcome onSignedIn={signIn} />;
  if (route.screen === "world") return <WorldScreen route={route} />;
  return <Home me={me} />;
}
