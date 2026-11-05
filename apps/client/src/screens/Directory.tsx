import type { WorldView } from "@airace/protocol";
import { Button } from "../components/Button.tsx";
import { Logo } from "../components/Logo.tsx";
import { Panel } from "../components/Panel.tsx";
import { makerOnline, sharedAis } from "../friends.ts";
import { rivalAis } from "../rivals.ts";
import { go } from "../route.ts";
import { useCards } from "../store/cards.ts";
import { usePlayground } from "../store/playground.ts";

export const MAKER_OFFLINE = "Its maker's computer is offline";

/**
 * The AI Directory (Stage 2): the AIs other players share with friends, and the rivals' AIs on
 * sale, each with "Try it" (a private chat in the Playground).
 */
export function Directory({ view }: { view: WorldView }) {
  const cards = useCards((s) => s.cards);
  const pick = usePlayground((s) => s.setVersion);
  const tryIt = (versionId: string) => {
    pick(versionId);
    go({ screen: "world", worldId: view.world.id, tab: "play" });
  };
  const shared = sharedAis(view);
  const rivals = rivalAis(view);
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold">AI Directory</h1>
        <p className="text-muted">
          AIs other players share with friends, and the rivals' AIs on sale. Your chats with them
          are private.
        </p>
      </header>
      <Panel title="Shared by players">
        {shared.length === 0 ? (
          <p className="text-muted">
            No one shares an AI yet. Share yours from the AI Lab's Versions.
          </p>
        ) : (
          <ul className="divide-y divide-border">
            {shared.map((ai) => {
              const online = makerOnline(view, cards, ai.makerUserId);
              return (
                <li key={ai.versionId} className="flex flex-wrap items-center gap-4 py-3">
                  <Logo logo={ai.company.logo} palette={ai.company.palette} size={40} />
                  <div className="mr-auto">
                    <p className="font-semibold">{ai.label}</p>
                    <p className="text-sm text-muted">
                      <span className="player-text">{ai.company.name}</span>
                      {" · "}
                      {ai.version.overall === null
                        ? "Not scored yet"
                        : `Overall score ${Math.round(ai.version.overall)}`}
                      {ai.version.personalityWords.length > 0 &&
                        ` · ${ai.version.personalityWords.join(", ")}`}
                    </p>
                    {!online && <p className="text-sm text-danger">{MAKER_OFFLINE}</p>}
                  </div>
                  <Button disabled={!online} onClick={() => tryIt(ai.versionId)}>
                    Try it
                  </Button>
                </li>
              );
            })}
          </ul>
        )}
      </Panel>
      {rivals.length > 0 && (
        <Panel title="Rivals' AIs on sale">
          <ul className="divide-y divide-border">
            {rivals.map((r) => (
              <li key={r.versionId} className="flex items-center gap-4 py-3">
                <p className="mr-auto">
                  <span className="font-semibold">{r.productName}</span>{" "}
                  <span className="text-sm text-muted">{r.companyName}</span>
                </p>
                <Button variant="secondary" onClick={() => tryIt(r.versionId)}>
                  Try it
                </Button>
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}
