import { ARCHETYPE_WORDS, type PublicCompany, STYLE_WORDS, type WorldView } from "@airace/protocol";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Module } from "../bench/Module.tsx";
import { PushButton } from "../bench/PushButton.tsx";
import { Logo } from "../components/Logo.tsx";
import { EmblemBadge } from "../emblem/EmblemBadge.tsx";
import { formatCount, formatMoney } from "../format.ts";
import { api } from "../net/api.ts";
import { rivalCompanies, shareOfWilling } from "../rivals.ts";
import { go } from "../route.ts";
import { OffersList } from "../spending/Licensing.tsx";
import { usePlayground } from "../store/playground.ts";
import { useWindows } from "../store/windows.ts";
import { useReadable } from "../theme/glass.ts";

/** Opens the Playground on a rival's AI (game-layer design §3 "Try it"). */
export function tryRival(view: WorldView, versionId: string): void {
  usePlayground.getState().setVersion(versionId);
  go({ screen: "world", worldId: view.world.id, tab: "play" });
}

function RivalCard({ view, company }: { view: WorldView; company: PublicCompany }) {
  const open = useWindows((s) => s.open);
  const readable = useReadable();
  const arch = company.rival?.archetype ?? null;
  const news = view.feed.filter((f) => f.kind === "news" && f.companyId === company.id).slice(-2);
  const offers = view.licenseOffers.filter((o) => o.licensorId === company.id);
  return (
    <motion.article
      aria-label={company.name}
      className={`module module--plate rival-plate ${arch ? "rival-plate--arch" : ""}`}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <header className="rival-plate__head">
        <button
          type="button"
          aria-label={`Open ${company.name}`}
          onClick={(e) => open({ kind: "rival", companyId: company.id }, e.currentTarget)}
          className="rival-plate__badge"
        >
          {arch ? (
            <EmblemBadge
              logo={company.logo}
              palette={company.palette}
              size={76}
              label={`${company.name}'s emblem`}
            />
          ) : (
            <Logo logo={company.logo} palette={company.palette} size={52} />
          )}
        </button>
        <div className="min-w-0">
          <h2 className="rival-plate__name" style={{ color: readable(company.palette.primary) }}>
            {company.name}
          </h2>
          <p className="rival-plate__kind">
            {arch
              ? `Arch-rival · ${ARCHETYPE_WORDS[arch]}`
              : STYLE_WORDS[company.rival?.style ?? "consumer"]}
          </p>
        </div>
      </header>
      <dl className="display rival-plate__stats">
        <div>
          <dt>Customers</dt>
          <dd className="dotface dotface--xs">
            {`${formatCount(company.customers)} (${shareOfWilling(view, company).toFixed(2)}%)`}
          </dd>
        </div>
        <div>
          <dt>Reputation</dt>
          <dd className="dotface dotface--xs">{company.reputation}</dd>
        </div>
        <div>
          <dt>Fans</dt>
          <dd className="dotface dotface--xs">{`${company.followers.toLocaleString("en-US")} followers`}</dd>
        </div>
      </dl>
      {company.products.length === 0 ? (
        <p className="rival-plate__note">No AI on sale yet.</p>
      ) : (
        <ul className="display rival-plate__products">
          {company.products.map((p) => (
            <li key={p.id} className="rival-plate__product">
              <span className="rival-plate__ai">
                <span className="rival-plate__ai-name">{p.name}</span>{" "}
                <span className="rival-plate__ai-what">
                  {p.status === "live"
                    ? `overall ${Math.round(p.version?.overall ?? 0)}${p.version?.personalityWords.length ? ` · ${p.version.personalityWords.join(", ")}` : ""}`
                    : "coming soon"}
                </span>
              </span>
              {p.status === "live" && p.version && (
                <span className="rival-plate__keys">
                  <PushButton
                    variant="secondary"
                    aria-label={`Try ${p.name}`}
                    onClick={() => tryRival(view, p.version?.id ?? "")}
                  >
                    Try it
                  </PushButton>
                  <PushButton
                    aria-label={`Battle ${p.name}`}
                    onClick={(e) =>
                      open({ kind: "newBattle", rivalProductId: p.id }, e.currentTarget)
                    }
                  >
                    Battle
                  </PushButton>
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
      {offers.length > 0 && (
        <div className="rival-plate__offers">
          <OffersList view={view} offers={offers} compact />
        </div>
      )}
      <p className="rival-plate__note">
        {company.prices.free ? "Free plan · " : ""}
        {company.prices.personal ? `Personal ${formatMoney(company.prices.personal)}/month · ` : ""}
        {company.prices.business ? `Business ${formatMoney(company.prices.business)}/seat` : ""}
      </p>
      {news.length > 0 && (
        <ul className="rival-plate__news">
          {news.map((n) => (
            <li key={n.id}>{n.kind === "news" ? n.headline : ""}</li>
          ))}
        </ul>
      )}
    </motion.article>
  );
}

/** Each rival's public profile (game-layer design §3): the arch-rival first. */
export function Rivals({ view }: { view: WorldView }) {
  const rivals = rivalCompanies(view);
  const [installed, setInstalled] = useState<boolean | null>(null);
  useEffect(() => {
    if (rivals.length > 0) return;
    api
      .setup()
      .then((s) => setInstalled(s.rivals))
      .catch(() => setInstalled(null));
  }, [rivals.length]);
  if (rivals.length === 0) {
    return (
      <Module label="Rivals" className="rivals-empty">
        <p>This world has no rivals yet.</p>
        {installed === false && (
          <p className="rival-plate__note">
            Their AIs aren't built on this computer yet. Building them takes about an hour and a
            half on the graphics card, and they join your worlds after that.
          </p>
        )}
      </Module>
    );
  }
  return (
    <div className="rivals-bench">
      {rivals.map((c) => (
        <RivalCard key={c.id} view={view} company={c} />
      ))}
    </div>
  );
}
