import type { VersionView, WorldView } from "@airace/protocol";
import { dateOfDay, getBalance } from "@airace/sim";
import { useEffect, useState } from "react";
import { Button } from "../../components/Button.tsx";
import { Panel } from "../../components/Panel.tsx";
import { capitalize, words } from "../../format.ts";
import { go } from "../../route.ts";
import { usePlan } from "../../store/plan.ts";
import { usePlayground } from "../../store/playground.ts";
import { useSocket } from "../../store/session.ts";
import { DiscoveryList } from "../../surprises/Discoveries.tsx";
import { LaunchDialog } from "../LaunchDialog.tsx";

function brainName(view: WorldView, v: VersionView): string {
  const b = getBalance(view.world.balance).brainSizes.find(
    (s) => s.nLayer === v.shape.nLayer && s.dModel === v.shape.dModel,
  );
  return b ? capitalize(b.id) : "Custom";
}

/** Every version of the company's AI, newest first. */
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
    return <p className="text-muted">No versions yet. Your first finished run becomes v1.</p>;
  return (
    <div className="space-y-3">
      {versions.map((v) => (
        <Panel key={v.id} data-testid={`version-${v.label}`}>
          <div className="flex flex-wrap items-center gap-4">
            <p className="text-2xl font-bold text-brand">{v.label}</p>
            <div className="mr-auto text-sm">
              <p>
                {brainName(view, v)} brain · read {words(v.trainedTokens)}
              </p>
              <p className="text-muted">
                Born {dateOfDay(v.createdDay)} ·{" "}
                {v.overall === null ? "Not scored yet" : `Overall score ${Math.round(v.overall)}`}
              </p>
            </div>
            <Button
              variant="secondary"
              onClick={() => {
                pick(v.id);
                go({ screen: "world", worldId: view.world.id, tab: "play" });
              }}
            >
              Chat
            </Button>
            {v.scores !== null && (
              <Button variant="secondary" onClick={() => setLaunching(v.id)}>
                Launch this AI
              </Button>
            )}
            <Button
              onClick={() => {
                update({ parentVersionId: v.id });
                onTrainFurther();
              }}
            >
              Train further from this
            </Button>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
            <Button variant="secondary" onClick={() => void share(v.id, !v.published)}>
              {v.published ? "Stop sharing" : "Share with friends"}
            </Button>
            {v.published && (
              <span className="text-muted">
                {tries[v.id] === undefined
                  ? "Shared in the AI Directory"
                  : `Tried ${tries[v.id]} ${tries[v.id] === 1 ? "time" : "times"} this week`}
              </span>
            )}
          </div>
          <div className="mt-3">
            <DiscoveryList
              discoveries={(view.me.company?.discoveries ?? []).filter((d) => d.versionId === v.id)}
            />
          </div>
        </Panel>
      ))}
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
