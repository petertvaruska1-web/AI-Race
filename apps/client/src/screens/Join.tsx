import { INVITE_ALPHABET, INVITE_LENGTH } from "@airace/protocol";
import { useEffect, useState } from "react";
import { Display } from "../bench/Display.tsx";
import { Keypad } from "../bench/Keypad.tsx";
import { Module } from "../bench/Module.tsx";
import { PushButton } from "../bench/PushButton.tsx";
import { FAIR_NOTICE } from "../fair.ts";
import { api, type InviteInfo, type Me } from "../net/api.ts";
import { Welcome } from "./Home.tsx";

/** Leaves the join link for the world itself (a fresh page at its address). */
function enterWorld(worldId: string): void {
  location.assign(`/#/w/${worldId}/company`);
}

/**
 * A friend's invite link, `/join/<code>` (Stage 2): a new player prints their ID badge first;
 * then the lab door's keypad shows the code filled in (C62), with the world's name and size, the
 * Fair mode notice for a Fair world, and "Join" or "Back". Another code typed on the keypad looks
 * up that world instead.
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
  const [typed, setTyped] = useState(code);
  const [info, setInfo] = useState<InviteInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!me) return;
    setInfo(null);
    setError(null);
    // The link's own code, or a whole code typed on the keypad.
    if (typed !== code && typed.length !== INVITE_LENGTH) return;
    let current = true;
    api
      .inviteInfo(typed)
      .then((i) => {
        if (!current) return;
        if (i.member) onJoined(i.worldId ?? "");
        setInfo(i);
      })
      .catch((err: Error) => {
        if (current) setError(err.message);
      });
    return () => {
      current = false;
    };
  }, [me, typed, code, onJoined]);

  if (!me) {
    return <Welcome onSignedIn={onSignedIn} intro="A friend invited you to their world." />;
  }

  async function join() {
    setBusy(true);
    setError(null);
    try {
      onJoined(await api.joinWorld(typed));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="door-frame door-frame--join">
      <Module weight="heavy" label="Join a friend" className="join-bench">
        <Keypad
          label="Invite code"
          alphabet={INVITE_ALPHABET}
          value={typed}
          maxLength={INVITE_LENGTH}
          onChange={setTyped}
        />
        {info && (
          <section aria-label="The world" className="join-bench__world">
            <span className="engrave">You're invited to</span>
            <Display>
              <p className="player-text m-0 text-xl font-bold">{info.name}</p>
            </Display>
            <Display>
              <p className="dotface dotface--sm m-0">
                {info.players} of {info.maxPlayers} players{info.fairMode ? " · Fair mode" : ""}
              </p>
            </Display>
            {info.fairMode && <p className="notice-plate">{FAIR_NOTICE}</p>}
          </section>
        )}
        {error && (
          <p role="alert" className="m-0 text-sm text-danger">
            {error}
          </p>
        )}
        <div className="flex gap-3">
          {info && (
            <PushButton onClick={join} disabled={busy}>
              Join
            </PushButton>
          )}
          <PushButton variant="secondary" onClick={() => location.assign("/")}>
            Back
          </PushButton>
        </div>
      </Module>
    </main>
  );
}
