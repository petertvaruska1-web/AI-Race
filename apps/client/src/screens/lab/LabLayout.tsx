import type { WorldView } from "@airace/protocol";
import { getBalance, previewPlan } from "@airace/sim";
import { useEffect, useState } from "react";
import { Button } from "../../components/Button.tsx";
import { formatMoney, gameDays } from "../../format.ts";
import { useCatalog } from "../../store/catalog.ts";
import { usePlan } from "../../store/plan.ts";
import { useSocket } from "../../store/session.ts";
import { DataStep } from "./DataStep.tsx";
import { DesignStep } from "./DesignStep.tsx";
import { PrepareStep } from "./PrepareStep.tsx";
import { TrainStep } from "./TrainStep.tsx";
import { VersionsStep } from "./VersionsStep.tsx";

export type LabStep = "data" | "prepare" | "design" | "train" | "versions";

const STEPS: { id: LabStep; label: string }[] = [
  { id: "data", label: "1 Data" },
  { id: "prepare", label: "2 Prepare" },
  { id: "design", label: "3 Design" },
  { id: "train", label: "4 Train" },
  { id: "versions", label: "5 Versions" },
];

/** The AI Lab: plan a run step by step, with the cost always in view, then watch it train. */
export function Lab({ view }: { view: WorldView }) {
  const socket = useSocket();
  const { catalog, error, load } = useCatalog();
  const { plan, worldId, reset } = usePlan();
  const company = view.me.company;
  const versions = company?.versions ?? [];
  const latest = versions.at(-1)?.id ?? null;
  const balance = getBalance(view.world.balance);
  const active = company?.runs.find((r) => r.status === "training" || r.status === "finalizing");
  const [step, setStep] = useState<LabStep>(active ? "train" : "data");

  useEffect(() => load(socket), [load, socket]);
  useEffect(() => {
    if (!plan || worldId !== view.world.id) reset(view.world.id, balance, latest);
  }, [plan, worldId, view.world.id, balance, latest, reset]);

  if (!plan || !company) return <p className="text-muted">Loading…</p>;
  const planning = step === "data" || step === "prepare" || step === "design";

  return (
    <div className="space-y-6">
      <nav className="flex flex-wrap gap-2" aria-label="Lab steps">
        {STEPS.map((s) => (
          <button
            key={s.id}
            type="button"
            aria-current={step === s.id ? "step" : undefined}
            onClick={() => setStep(s.id)}
            className={`rounded-full px-4 py-1.5 text-sm ${step === s.id ? "bg-brand text-on-brand" : "bg-surface text-muted hover:text-text"}`}
          >
            {s.label}
          </button>
        ))}
      </nav>
      {planning && !catalog && (
        <p className="text-muted">{error ?? "Loading the training data…"}</p>
      )}
      {step === "data" && catalog && <DataStep plan={plan} catalog={catalog} />}
      {step === "prepare" && catalog && (
        <PrepareStep plan={plan} catalog={catalog} balance={balance} />
      )}
      {step === "design" && <DesignStep plan={plan} balance={balance} versions={versions} />}
      {step === "train" && <TrainStep view={view} />}
      {step === "versions" && <VersionsStep view={view} onTrainFurther={() => setStep("data")} />}
      {planning && <SummaryBar view={view} onStarted={() => setStep("train")} />}
    </div>
  );
}

/** The plan's live price and game time, and the button that starts it. */
function SummaryBar({ view, onStarted }: { view: WorldView; onStarted: () => void }) {
  const socket = useSocket();
  const plan = usePlan((s) => s.plan);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const company = view.me.company;
  if (!plan || !company) return null;
  const parent = company.versions.find((v) => v.id === plan.parentVersionId)?.shape ?? null;
  const preview = previewPlan(getBalance(view.world.balance), parent, plan);
  const active = company.runs.some((r) => r.status === "training" || r.status === "finalizing");
  const blocked = !preview.ok
    ? preview.reason
    : active
      ? "Your AI is already training."
      : company.cash < preview.preview.cost
        ? `Not enough cash: this costs ${formatMoney(preview.preview.cost)} and you have ${formatMoney(company.cash)}.`
        : null;

  async function start() {
    if (!plan) return;
    setBusy(true);
    setError(null);
    const r = await socket.command(view.world.id, { type: "startTraining", plan });
    setBusy(false);
    if (r.ok) onStarted();
    else setError(r.reason);
  }

  return (
    <div className="sticky bottom-4 flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-brand bg-surface p-4 shadow-lg">
      <div>
        {preview.ok ? (
          <p data-testid="plan-cost" className="text-lg font-semibold">
            {formatMoney(preview.preview.cost)}{" "}
            <span className="text-sm font-normal text-muted">
              · {gameDays(preview.preview.durationDays)}
            </span>
          </p>
        ) : null}
        {(blocked ?? error) && (
          <p role="alert" className="text-sm text-danger">
            {blocked ?? error}
          </p>
        )}
      </div>
      <Button onClick={start} disabled={busy || blocked !== null} title={blocked ?? undefined}>
        Start training
      </Button>
    </div>
  );
}
