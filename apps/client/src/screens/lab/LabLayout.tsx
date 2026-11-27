import type { WorldView } from "@airace/protocol";
import { getBalance, previewPlan } from "@airace/sim";
import { useEffect, useState } from "react";
import { PushButton } from "../../bench/PushButton.tsx";
import { StepDial } from "../../bench/StepDial.tsx";
import { labLimit, NO_CARD, realMinutes } from "../../cards/Cards.tsx";
import { lentToTraining } from "../../components/Computers.tsx";
import { formatMoney, gameDays } from "../../format.ts";
import { useCards } from "../../store/cards.ts";
import { useCatalog } from "../../store/catalog.ts";
import { usePlan } from "../../store/plan.ts";
import { useSocket } from "../../store/session.ts";
import { DataStep } from "./DataStep.tsx";
import { DesignStep } from "./DesignStep.tsx";
import { PrepareStep } from "./PrepareStep.tsx";
import { TrainStep } from "./TrainStep.tsx";
import { VersionsStep } from "./VersionsStep.tsx";

export type LabStep = "data" | "prepare" | "design" | "train" | "versions";

const STEPS: { value: LabStep; label: string }[] = [
  { value: "data", label: "1 Data" },
  { value: "prepare", label: "2 Prepare" },
  { value: "design", label: "3 Design" },
  { value: "train", label: "4 Train" },
  { value: "versions", label: "5 Versions" },
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
  const cards = useCards((s) => s.cards);
  const cardLimit = useCards((s) => s.limit);
  const limits = labLimit(view, cards, cardLimit, balance);

  useEffect(() => load(socket), [load, socket]);
  useEffect(() => {
    if (!plan || worldId !== view.world.id) reset(view.world.id, balance, latest);
  }, [plan, worldId, view.world.id, balance, latest, reset]);

  if (!plan || !company) return <p className="text-muted">Loading…</p>;
  const planning = step === "data" || step === "prepare" || step === "design";

  return (
    <div className="lab-bench">
      <StepDial
        legend="Lab steps"
        steps={STEPS}
        value={step}
        onChange={setStep}
        className="lab-bench__steps"
      />
      {planning && !catalog && (
        <p className="text-muted">{error ?? "Loading the training data…"}</p>
      )}
      {step === "data" && catalog && <DataStep plan={plan} catalog={catalog} view={view} />}
      {step === "prepare" && catalog && (
        <PrepareStep plan={plan} catalog={catalog} balance={balance} />
      )}
      {step === "design" && (
        <DesignStep
          plan={plan}
          balance={balance}
          versions={versions}
          lent={lentToTraining(view)}
          loved={view.me.company?.loved ?? 0}
          maxParams={limits.maxParams}
          limitNote={limits.note}
        />
      )}
      {step === "train" && <TrainStep view={view} />}
      {step === "versions" && <VersionsStep view={view} onTrainFurther={() => setStep("data")} />}
      {planning && <SummaryBar view={view} onStarted={() => setStep("train")} />}
    </div>
  );
}

/** The plan's live price and game time, and the button that starts it. */
function SummaryBar({ view, onStarted }: { view: WorldView; onStarted: () => void }) {
  const socket = useSocket();
  const cards = useCards((s) => s.cards);
  const limit = useCards((s) => s.limit);
  const plan = usePlan((s) => s.plan);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const company = view.me.company;
  if (!plan || !company) return null;
  const parent = company.versions.find((v) => v.id === plan.parentVersionId)?.shape ?? null;
  const lent = lentToTraining(view);
  const loved = company.loved;
  const lessons = company.lessons;
  // Research reads more for the money; the Data team makes checks cheaper (as the server prices).
  const teams = {
    research: company.departments.research.strength,
    data: company.departments.data.strength,
  };
  const b = getBalance(view.world.balance);
  const preview = previewPlan(b, parent, plan, lent, loved, lessons, teams);
  const rented = previewPlan(b, parent, plan, 0, loved, lessons, teams);
  const saved =
    lent > 0 && preview.ok && rented.ok
      ? {
          days: rented.preview.durationDays - preview.preview.durationDays,
          dollars: rented.preview.cost - preview.preview.cost,
        }
      : null;
  const active = company.runs.some((r) => r.status === "training" || r.status === "finalizing");
  const limits = labLimit(view, cards, limit);
  const minutes = preview.ok
    ? realMinutes(preview.preview.tokens, preview.preview.params, limits.tokensPerSecond)
    : null;
  const blocked = !preview.ok
    ? preview.reason
    : limits.noCard
      ? NO_CARD
      : limits.maxParams !== null && preview.preview.params > limits.maxParams
        ? limits.note
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
    <div className="module module--plate lab-summary">
      {preview.ok && (
        <div className="display lab-summary__cost">
          <span data-testid="plan-cost" className="dotface dotface--md">
            {formatMoney(preview.preview.cost)}
          </span>
          <span className="lab-summary__days">{gameDays(preview.preview.durationDays)}</span>
        </div>
      )}
      <div className="lab-summary__notes">
        {minutes !== null && (
          <p data-testid="real-minutes" className="lab-summary__note">
            {`About ${minutes} ${minutes === 1 ? "minute" : "minutes"} ${limits.where}.`}
          </p>
        )}
        {saved && (saved.days > 0 || saved.dollars > 0) && (
          <p className="lab-summary__note text-good">
            Your data centers make this run {gameDays(saved.days)} faster and{" "}
            {formatMoney(saved.dollars)} cheaper.
          </p>
        )}
        {(blocked ?? error) && (
          <p role="alert" className="lab-summary__note text-danger">
            {blocked ?? error}
          </p>
        )}
      </div>
      <PushButton onClick={start} disabled={busy || blocked !== null} title={blocked ?? undefined}>
        Start training
      </PushButton>
    </div>
  );
}
