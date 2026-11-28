import type { WorldView } from "@airace/protocol";
import type { ReactNode } from "react";
import { Display } from "../bench/Display.tsx";
import { Module } from "../bench/Module.tsx";
import { PushButton } from "../bench/PushButton.tsx";
import { Logo } from "../components/Logo.tsx";
import { makerOnline, sharedAis } from "../friends.ts";
import { rivalAis } from "../rivals.ts";
import { go } from "../route.ts";
import { OffersList } from "../spending/Licensing.tsx";
import { useCards } from "../store/cards.ts";
import { usePlayground } from "../store/playground.ts";
import { useWindows } from "../store/windows.ts";

export const MAKER_OFFLINE = "Its maker's computer is offline";

/** One catalog of the Directory: a module with its list in a display, or a line when empty. */
function Catalog({
  title,
  list,
  hint,
  empty,
  children,
}: {
  title: string;
  list: string;
  hint: string;
  empty: string | null;
  children: ReactNode;
}) {
  return (
    <Module label={title} hint={hint} aria-label={title} className="catalog">
      {empty !== null ? (
        <p className="catalog__empty">{empty}</p>
      ) : (
        <Display scroll label={list} className="catalog__list">
          {children}
        </Display>
      )}
    </Module>
  );
}

/**
 * The AI Directory (Stage 2): the AIs other players share with friends, the license offers, the
 * players' AIs you can ask to license, and the rivals' AIs on sale, each a catalog on the bench.
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
    <div className="directory-bench">
      <Catalog
        title="Shared by players"
        list="Shared AIs"
        hint="AIs other players share with friends. Your chats with them are private."
        empty={
          shared.length === 0
            ? "No one shares an AI yet. Share yours from the AI Lab's Versions."
            : null
        }
      >
        <ul>
          {shared.map((ai) => {
            const online = makerOnline(view, cards, ai.makerUserId);
            return (
              <li key={ai.versionId} className="catalog__item">
                <Logo logo={ai.company.logo} palette={ai.company.palette} size={36} />
                <div className="catalog__what">
                  <p className="catalog__name">{ai.label}</p>
                  <p className="catalog__detail">
                    <span className="player-text">{ai.company.name}</span>
                    {" · "}
                    {ai.version.overall === null
                      ? "Not scored yet"
                      : `Overall score ${Math.round(ai.version.overall)}`}
                    {ai.version.personalityWords.length > 0 &&
                      ` · ${ai.version.personalityWords.join(", ")}`}
                  </p>
                  {!online && <p className="catalog__warn">{MAKER_OFFLINE}</p>}
                </div>
                <PushButton disabled={!online} onClick={() => tryIt(ai.versionId)}>
                  Try it
                </PushButton>
              </li>
            );
          })}
        </ul>
      </Catalog>
      <Catalog
        title="License offers"
        list="Offers"
        hint="AIs other companies rent out: one answers your customers in the groups you pick, for a price per customer a month."
        empty={offers.length === 0 ? "No one offers an AI for license yet." : null}
      >
        <OffersList view={view} offers={offers} />
      </Catalog>
      <Catalog
        title="Ask a player"
        list="Players' AIs"
        hint="Other players' AIs on sale. Ask to license one and its maker decides."
        empty={playersOnSale.length === 0 ? "No other player has an AI on sale yet." : null}
      >
        <ul>
          {playersOnSale.map((x) => (
            <li key={x.product.id} className="catalog__item">
              <Logo logo={x.company.logo} palette={x.company.palette} size={36} />
              <div className="catalog__what">
                <p className="catalog__name">{x.product.name}</p>
                <p className="catalog__detail">
                  <span className="player-text">{x.company.name}</span>
                </p>
              </div>
              <PushButton
                variant="secondary"
                aria-label={`Ask to license ${x.product.name}`}
                onClick={(e) => open({ kind: "propose", versionId: x.versionId }, e.currentTarget)}
              >
                Ask to license
              </PushButton>
            </li>
          ))}
        </ul>
      </Catalog>
      <Catalog
        title="Rivals' AIs on sale"
        list="Rivals' AIs"
        hint="Try a rival's AI in the Playground before you battle it."
        empty={rivals.length === 0 ? "No rival has an AI on sale yet." : null}
      >
        <ul>
          {rivals.map((r) => (
            <li key={r.versionId} className="catalog__item">
              <div className="catalog__what">
                <p className="catalog__name">{r.productName}</p>
                <p className="catalog__detail">{r.companyName}</p>
              </div>
              <PushButton variant="secondary" onClick={() => tryIt(r.versionId)}>
                Try it
              </PushButton>
            </li>
          ))}
        </ul>
      </Catalog>
    </div>
  );
}
