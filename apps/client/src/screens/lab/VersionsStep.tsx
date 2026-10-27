import type { VersionView, WorldView } from "@airace/protocol";
import { dateOfDay, getBalance } from "@airace/sim";
import { Button } from "../../components/Button.tsx";
import { Card } from "../../components/Card.tsx";
import { capitalize, words } from "../../format.ts";
import { go } from "../../route.ts";
import { usePlan } from "../../store/plan.ts";
import { usePlayground } from "../../store/playground.ts";

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
  if (versions.length === 0)
    return <p className="text-muted">No versions yet. Your first finished run becomes v1.</p>;
  return (
    <div className="space-y-3">
      {versions.map((v) => (
        <Card key={v.id} data-testid={`version-${v.label}`}>
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
            <Button
              onClick={() => {
                update({ parentVersionId: v.id });
                onTrainFurther();
              }}
            >
              Train further from this
            </Button>
          </div>
        </Card>
      ))}
    </div>
  );
}
