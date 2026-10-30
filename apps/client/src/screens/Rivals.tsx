import { ARCHETYPE_WORDS, type PublicCompany, STYLE_WORDS, type WorldView } from "@airace/protocol";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Button } from "../components/Button.tsx";
import { Logo } from "../components/Logo.tsx";
import { Panel } from "../components/Panel.tsx";
import { formatCount, formatMoney } from "../format.ts";
import { api } from "../net/api.ts";
import { rivalCompanies, shareOfWilling } from "../rivals.ts";
import { go } from "../route.ts";
import { usePlayground } from "../store/playground.ts";
import { useWindows } from "../store/windows.ts";

/** Opens the Playground on a rival's AI (game-layer design §3 "Try it"). */
export function tryRival(view: WorldView, versionId: string): void {
  usePlayground.getState().setVersion(versionId);
  go({ screen: "world", worldId: view.world.id, tab: "play" });
}

function RivalCard({ view, company }: { view: WorldView; company: PublicCompany }) {
  const open = useWindows((s) => s.open);
  const arch = company.rival?.archetype ?? null;
  const news = view.feed.filter((f) => f.kind === "news" && f.companyId === company.id).slice(-2);
  return (
    <motion.article
      aria-label={company.name}
      className={`panel p-5 ${arch ? "md:col-span-2 ring-1 ring-danger/40" : ""}`}
      style={{ borderColor: company.palette.primary }}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <header className="flex items-center gap-4">
        <button
          type="button"
          aria-label={`Open ${company.name}`}
          onClick={(e) => open({ kind: "rival", companyId: company.id }, e.currentTarget)}
          className="rounded-2xl transition hover:scale-105"
        >
          <Logo logo={company.logo} palette={company.palette} size={arch ? 72 : 56} />
        </button>
        <div className="min-w-0">
          <h2 className="font-display text-xl font-bold" style={{ color: company.palette.primary }}>
            {company.name}
          </h2>
          <p className="text-sm text-muted">
            {arch
              ? `Arch-rival · ${ARCHETYPE_WORDS[arch]}`
              : STYLE_WORDS[company.rival?.style ?? "consumer"]}
          </p>
        </div>
      </header>
      <dl className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <div>
          <dt className="text-muted">Customers</dt>
          <dd className="font-semibold">
            {formatCount(company.customers)} ({shareOfWilling(view, company).toFixed(2)}%)
          </dd>
        </div>
        <div>
          <dt className="text-muted">Reputation</dt>
          <dd className="font-semibold">{company.reputation}</dd>
        </div>
        <div>
          <dt className="text-muted">Fans</dt>
          <dd className="font-semibold">{`${company.followers.toLocaleString("en-US")} followers`}</dd>
        </div>
      </dl>
      <ul className="mt-4 space-y-2">
        {company.products.map((p) => (
          <li key={p.id} className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <span>
              <span className="font-semibold">{p.name}</span>{" "}
              <span className="text-muted">
                {p.status === "live"
                  ? `overall ${Math.round(p.version?.overall ?? 0)}${p.version?.personalityWords.length ? ` · ${p.version.personalityWords.join(", ")}` : ""}`
                  : "coming soon"}
              </span>
            </span>
            {p.status === "live" && p.version && (
              <span className="flex gap-2">
                <Button
                  variant="secondary"
                  aria-label={`Try ${p.name}`}
                  onClick={() => tryRival(view, p.version?.id ?? "")}
                >
                  Try it
                </Button>
                <Button
                  aria-label={`Battle ${p.name}`}
                  onClick={(e) =>
                    open({ kind: "newBattle", rivalProductId: p.id }, e.currentTarget)
                  }
                >
                  Battle
                </Button>
              </span>
            )}
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-muted">
        {company.prices.free ? "Free plan · " : ""}
        {company.prices.personal ? `Personal ${formatMoney(company.prices.personal)}/month · ` : ""}
        {company.prices.business ? `Business ${formatMoney(company.prices.business)}/seat` : ""}
      </p>
      {news.length > 0 && (
        <ul className="mt-3 space-y-1 border-t border-border pt-3 text-xs text-muted">
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
      <Panel title="Rivals">
        <p>This world has no rivals yet.</p>
        {installed === false && (
          <p className="mt-2 text-sm text-muted">
            Their AIs aren't built on this computer yet. Building them takes about an hour and a
            half on the graphics card, and they join your worlds after that.
          </p>
        )}
      </Panel>
    );
  }
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {rivals.map((c) => (
        <RivalCard key={c.id} view={view} company={c} />
      ))}
    </div>
  );
}
