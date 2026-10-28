import type { WorldView } from "@airace/protocol";
import { Button } from "../components/Button.tsx";
import { Panel } from "../components/Panel.tsx";
import { go } from "../route.ts";

// The company screen's first draft (Stage 1's dashboard); Task 12 replaces it.

const RUN_STATUS: Record<string, string> = {
  training: "Training",
  finalizing: "Finishing up",
  completed: "Done",
  failed: "Failed",
};

export function Dashboard({ view }: { view: WorldView }) {
  const company = view.me.company;
  if (!company) return null;
  const run = company.runs.at(-1);
  const latest = company.versions.at(-1);
  const lab = () => go({ screen: "world", worldId: view.world.id, tab: "lab" });
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Panel title="Training">
        {run ? (
          <div className="space-y-2">
            <p>
              <span className="font-semibold">{RUN_STATUS[run.status]}</span>
              {run.status === "training" && !run.computeStarted && (
                <span className="text-muted"> · waiting for a lab computer</span>
              )}
            </p>
            {(run.status === "training" || run.status === "finalizing") && (
              <progress
                className="h-2 w-full accent-[var(--brand)]"
                aria-label="Training progress"
                value={run.progress}
                max={1}
              />
            )}
            <Button variant="secondary" onClick={lab}>
              Open the AI Lab
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-muted">
              You have no AI yet. Plan and train your first one in the AI Lab.
            </p>
            <Button onClick={lab}>Train your first AI</Button>
          </div>
        )}
      </Panel>
      <Panel title="Your latest AI">
        {latest ? (
          <div className="space-y-1">
            <p className="text-2xl font-bold text-brand">{latest.label}</p>
            <p className="text-sm text-muted">
              {latest.overall === null
                ? "Not scored yet"
                : `Overall score ${Math.round(latest.overall)}`}
            </p>
            {latest.personalityWords.length > 0 && (
              <p className="text-sm">{latest.personalityWords.join(", ")}</p>
            )}
          </div>
        ) : (
          <p className="text-muted">Nothing trained yet.</p>
        )}
      </Panel>
    </div>
  );
}
