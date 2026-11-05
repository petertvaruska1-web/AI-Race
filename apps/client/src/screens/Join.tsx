import { useEffect, useState } from "react";
import { Button } from "../components/Button.tsx";
import { Panel } from "../components/Panel.tsx";
import { FAIR_NOTICE } from "../fair.ts";
import { api, type InviteInfo, type Me } from "../net/api.ts";
import { Welcome } from "./Home.tsx";

/** Leaves the join link for the world itself (a fresh page at its address). */
function enterWorld(worldId: string): void {
  location.assign(`/#/w/${worldId}/company`);
}

/**
 * A friend's invite link, `/join/<code>` (Stage 2): a new player picks a name first; then the
 * world's name and size show, with the Fair mode notice for a Fair world, and "Join" or "Back".
 */
export function Join({
  code,
  me,
  onSignedIn,
  onJoined = enterWorld,
}: {
  code: string;
  me: Me | null;
  onSignedIn: (me: Me) => void;
  onJoined?: (worldId: string) => void;
}) {
  const [info, setInfo] = useState<InviteInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!me) return;
    api
      .inviteInfo(code)
      .then((i) => {
        if (i.member) onJoined(i.worldId ?? "");
        setInfo(i);
      })
      .catch((err: Error) => setError(err.message));
  }, [me, code, onJoined]);

  if (!me) {
    return <Welcome onSignedIn={onSignedIn} intro="A friend invited you to their world." />;
  }

  async function join() {
    setBusy(true);
    setError(null);
    try {
      onJoined(await api.joinWorld(code));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto mt-24 max-w-md space-y-4 px-6">
      <p className="text-sm text-muted">You're invited to</p>
      {info && (
        <Panel>
          <h1 className="player-text mb-1 text-2xl font-bold">{info.name}</h1>
          <p className="mb-4 text-sm text-muted">
            {info.players} of {info.maxPlayers} players
            {info.fairMode ? " · Fair mode" : ""}
          </p>
          {info.fairMode && (
            <p className="mb-4 rounded-lg border border-border p-3 text-sm">{FAIR_NOTICE}</p>
          )}
          <div className="flex gap-3">
            <Button onClick={join} disabled={busy}>
              Join
            </Button>
            <Button variant="secondary" onClick={() => location.assign("/")}>
              Back
            </Button>
          </div>
        </Panel>
      )}
      {error && (
        <p role="alert" className="text-sm text-danger">
          {error}
        </p>
      )}
    </main>
  );
}
