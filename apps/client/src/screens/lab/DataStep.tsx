import { DATASET_IDS, type PublicCatalog, type TrainingPlan } from "@airace/protocol";
import { Panel } from "../../components/Panel.tsx";
import { percent, words } from "../../format.ts";
import { usePlan } from "../../store/plan.ts";

/** What the AI reads: one card per dataset, with a slider for its share of the reading. */
export function DataStep({ plan, catalog }: { plan: TrainingPlan; catalog: PublicCatalog }) {
  const setWeight = usePlan((s) => s.setWeight);
  const total = DATASET_IDS.reduce((sum, id) => sum + plan.mix[id], 0);
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted">
        Choose what your AI reads. It learns the patterns in whatever you give it, so the mix shapes
        what it becomes good at.
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        {catalog.datasets.map((d) => {
          const weight = plan.mix[d.id];
          return (
            <Panel key={d.id} data-testid={`dataset-${d.id}`}>
              <div className="mb-2 flex items-baseline justify-between gap-2">
                <h3 className="text-lg font-semibold">{d.title}</h3>
                <span className="text-sm font-semibold text-brand">
                  {total > 0 ? percent(weight / total) : "0%"}
                </span>
              </div>
              <p className="text-sm text-muted">{d.blurb}</p>
              <p className="mt-1 text-xs text-muted">{words(d.tokens)}</p>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={weight}
                aria-label={`${d.title} share`}
                onChange={(e) => setWeight(d.id, Number(e.target.value))}
                className="mt-3 w-full accent-[var(--brand)]"
              />
              <details className="mt-2 text-sm">
                <summary className="cursor-pointer text-muted hover:text-text">
                  See examples
                </summary>
                <ul className="mt-2 space-y-2">
                  {d.samples.map((sample) => (
                    <li key={sample} className="player-text rounded-lg bg-surface-2 p-3 text-xs">
                      {sample}
                    </li>
                  ))}
                </ul>
              </details>
            </Panel>
          );
        })}
      </div>
    </div>
  );
}
