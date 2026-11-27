import type { VersionView, WorldView } from "@airace/protocol";
import { dateOfDay, getBalance } from "@airace/sim";
import { useEffect, useState } from "react";
import { Display } from "../../bench/Display.tsx";
import { PushButton } from "../../bench/PushButton.tsx";
import { Core } from "../../core/Core.tsx";
import { capitalize, words } from "../../format.ts";
import { go } from "../../route.ts";
import { usePlan } from "../../store/plan.ts";
import { usePlayground } from "../../store/playground.ts";
import { useSocket } from "../../store/session.ts";
import { useWindows } from "../../store/windows.ts";
import { DiscoveryList } from "../../surprises/Discoveries.tsx";
import { LaunchDialog } from "../LaunchDialog.tsx";

function brainName(view: WorldView, v: VersionView): string {
  const b = getBalance(view.world.balance).brainSizes.find(
    (s) => s.nLayer === v.shape.nLayer && s.dModel === v.shape.dModel,
  );
  return b ? capitalize(b.id) : "Custom";
}

/**
 * Every version of the company's AI, newest first, as cartridges on a rack (design §10): each
 * with its small core (opening its AI), its label and scores, and its keys.
 */
export function VersionsStep({
  view,
  onTrainFurther,
}: {
  view: WorldView;
  onTrainFurther: () => void;
}) {
  const versions = [...(view.me.company?.versions ?? [])].reverse();
  const update = usePlan((s) => s.update);
  const pick = usePlayground((s) => s.setVersion);
  const [launching, setLaunching] = useState<string | null>(null);
  const open = useWindows((s) => s.open);
  const palette = view.companies.find((c) => c.id === view.me.company?.id)?.palette;
  const socket = useSocket();
  const [tries, setTries] = useState<Record<string, number>>({});
  const [shareError, setShareError] = useState<string | null>(null);
  const sharedKey = versions
    .filter((v) => v.published)
    .map((v) => v.id)
    .join(",");
  useEffect(() => {
    if (!sharedKey) return;
    socket
      .query<{ versions: { versionId: string; tries7d: number }[] }>({
        kind: "versionTries",
        worldId: view.world.id,
      })
      .then((r) => setTries(Object.fromEntries(r.versions.map((x) => [x.versionId, x.tries7d]))))
      .catch(() => setTries({}));
  }, [socket, view.world.id, sharedKey]);
  const share = async (versionId: string, published: boolean) => {
    setShareError(null);
    const r = await socket.command(view.world.id, { type: "publishVersion", versionId, published });
    if (!r.ok) setShareError(r.reason);
  };
  if (versions.length === 0)
    return <p className="lab-note">No versions yet. Your first finished run becomes v1.</p>;
  return (
    <div className="lab-versions">
      <Display scroll label="Your versions" className="lab-rack">
        <ul className="lab-rack__list">
          {versions.map((v) => (
            <li key={v.id} data-testid={`version-${v.label}`} className="lab-cartridge">
              <div className="lab-cartridge__head">
                <button
                  type="button"
                  aria-label={`Open ${v.label}`}
                  className="lab-cartridge__core"
                  onClick={(e) => open({ kind: "ai", versionId: v.id }, e.currentTarget)}
                >
                  {palette && (
                    <Core
                      size={72}
                      still
                      label={`Your AI ${v.label}`}
                      input={{
                        overall: v.overall,
                        scores: v.scores,
                        traits: v.traits,
                        trainedTokens: v.trainedTokens,
                        palette,
                        training: null,
                      }}
                    />
                  )}
                </button>
                <div className="lab-cartridge__id">
                  <span className="dotface dotface--md">{v.label}</span>
                  <p className="lab-cartridge__meta">
                    {brainName(view, v)} brain · read {words(v.trainedTokens)}
                  </p>
                  <p className="lab-cartridge__meta">
                    Born {dateOfDay(v.createdDay)} ·{" "}
                    {v.overall === null
                      ? "Not scored yet"
                      : `Overall score ${Math.round(v.overall)}`}
                  </p>
                </div>
              </div>
              <div className="lab-cartridge__keys">
                <PushButton
                  variant="secondary"
                  onClick={() => {
                    pick(v.id);
                    go({ screen: "world", worldId: view.world.id, tab: "play" });
                  }}
                >
                  Chat
                </PushButton>
                {v.scores !== null && (
                  <PushButton variant="secondary" onClick={() => setLaunching(v.id)}>
                    Launch this AI
                  </PushButton>
                )}
                <PushButton
                  onClick={() => {
                    update({ parentVersionId: v.id });
                    onTrainFurther();
                  }}
                >
                  Train further from this
                </PushButton>
                <PushButton variant="secondary" onClick={() => void share(v.id, !v.published)}>
                  {v.published ? "Stop sharing" : "Share with friends"}
                </PushButton>
              </div>
              {v.published && (
                <p className="lab-cartridge__meta">
                  {tries[v.id] === undefined
                    ? "Shared in the AI Directory"
                    : `Tried ${tries[v.id]} ${tries[v.id] === 1 ? "time" : "times"} this week`}
                </p>
              )}
              <DiscoveryList
                discoveries={(view.me.company?.discoveries ?? []).filter(
                  (d) => d.versionId === v.id,
                )}
              />
            </li>
          ))}
        </ul>
      </Display>
      {shareError && (
        <p role="alert" className="text-sm text-danger">
          {shareError}
        </p>
      )}
      {launching && (
        <LaunchDialog view={view} versionId={launching} onClose={() => setLaunching(null)} />
      )}
    </div>
  );
}
