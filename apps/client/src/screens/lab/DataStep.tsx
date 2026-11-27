import { MIX_IDS, type PublicCatalog, type TrainingPlan, type WorldView } from "@airace/protocol";
import { useState } from "react";
import { Display } from "../../bench/Display.tsx";
import { Fader } from "../../bench/Fader.tsx";
import { Module } from "../../bench/Module.tsx";
import { percent, words } from "../../format.ts";
import { usePlan } from "../../store/plan.ts";
import { LessonsCard } from "../../surprises/Lessons.tsx";
import { NotebookChannel } from "../../surprises/Notebook.tsx";

/**
 * What the AI reads (design §10, the AI Lab): a mixing desk with one fader per dataset, each with
 * its share in a display above it and a level meter beside it, your notebook as one more channel,
 * and one display under the desk showing the chosen channel's examples as text.
 */
export function DataStep({
  plan,
  catalog,
  view,
}: {
  plan: TrainingPlan;
  catalog: PublicCatalog;
  view?: WorldView;
}) {
  const setWeight = usePlan((s) => s.setWeight);
  const total = MIX_IDS.reduce((sum, id) => sum + plan.mix[id], 0);
  const [shown, setShown] = useState(catalog.datasets[0]?.id ?? null);
  const example = catalog.datasets.find((d) => d.id === shown) ?? null;
  return (
    <div className="lab-data">
      <Module
        label="What your AI reads"
        weight="heavy"
        hint="It learns the patterns in whatever you give it, so the mix shapes what it becomes good at."
        className="lab-desk"
      >
        <div className="lab-desk__channels">
          {catalog.datasets.map((d) => {
            const share = total > 0 ? plan.mix[d.id] / total : 0;
            return (
              <div key={d.id} className="lab-channel" data-testid={`dataset-${d.id}`}>
                <Fader
                  label={`${d.title} share`}
                  value={plan.mix[d.id]}
                  min={0}
                  max={100}
                  onChange={(v) => setWeight(d.id, v)}
                  level={share}
                  readout={total > 0 ? percent(share) : "0%"}
                />
                <h3 className="engrave lab-channel__title">{d.title}</h3>
                <p className="lab-channel__size">{words(d.tokens)}</p>
                <button
                  type="button"
                  aria-pressed={shown === d.id}
                  className={`keycap lab-channel__key ${shown === d.id ? "keycap--on" : ""}`}
                  onClick={() => setShown(d.id)}
                >
                  <span className="keycap__lamp" aria-hidden="true" />
                  See examples
                </button>
              </div>
            );
          })}
          {view && <NotebookChannel view={view} total={total} />}
        </div>
      </Module>
      <div className="lab-data__below">
        <Module
          label={example ? `${example.title}: examples` : "Examples"}
          className="lab-examples"
        >
          {example && (
            <Display scroll label="Examples" className="lab-examples__display">
              <p className="lab-examples__blurb">{example.blurb}</p>
              <ul>
                {example.samples.map((sample) => (
                  <li key={sample} className="player-text lab-examples__sample">
                    {sample}
                  </li>
                ))}
              </ul>
            </Display>
          )}
        </Module>
        {view && <LessonsCard view={view} />}
      </div>
    </div>
  );
}
