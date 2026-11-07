import type { WorldView } from "@airace/protocol";
import { Button } from "../components/Button.tsx";
import { Logo } from "../components/Logo.tsx";
import { Panel } from "../components/Panel.tsx";
import { makerOnline, sharedAis } from "../friends.ts";
import { rivalAis } from "../rivals.ts";
import { go } from "../route.ts";
import { OffersList } from "../spending/Licensing.tsx";
import { useCards } from "../store/cards.ts";
import { usePlayground } from "../store/playground.ts";
import { useWindows } from "../store/windows.ts";

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
  const open = useWindows((s) => s.open);
  const shared = sharedAis(view);
  const rivals = rivalAis(view);
  const mine = view.me.company?.id;
  const offers = view.licenseOffers.filter((o) => o.licensorId !== mine);
  // Players' AIs on sale, which you can ask to license (C44).
  const playersOnSale = view.companies
    .filter((c) => c.id !== mine && !c.rival)
    .flatMap((c) =>
      c.products
        .filter((p) => p.status === "live" && p.version)
        .map((p) => ({ company: c, product: p, versionId: p.version?.id ?? "" })),
    );
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
      <Panel
        title="License offers"
        aria-label="License offers"
        hint="AIs other companies rent out: one answers your customers in the groups you pick, for a price per customer a month."
      >
        {offers.length === 0 ? (
          <p className="text-muted">No one offers an AI for license yet.</p>
        ) : (
          <OffersList view={view} offers={offers} />
        )}
      </Panel>
      {playersOnSale.length > 0 && (
        <Panel title="Ask a player">
          <ul className="divide-y divide-border">
            {playersOnSale.map((x) => (
              <li key={x.product.id} className="flex flex-wrap items-center gap-4 py-3">
                <p className="mr-auto">
                  <span className="font-semibold">{x.product.name}</span>{" "}
                  <span className="player-text text-sm text-muted">{x.company.name}</span>
                </p>
                <Button
                  variant="secondary"
                  aria-label={`Ask to license ${x.product.name}`}
                  onClick={(e) =>
                    open({ kind: "propose", versionId: x.versionId }, e.currentTarget)
                  }
                >
                  Ask to license
                </Button>
              </li>
            ))}
          </ul>
        </Panel>
      )}
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
