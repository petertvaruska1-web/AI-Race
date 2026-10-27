import { type PublicCatalog, prepKey, type TrainingPlan } from "@airace/protocol";
import type { Balance } from "@airace/sim";
import type { ReactNode } from "react";
import { Card } from "../../components/Card.tsx";
import { formatMoney, words } from "../../format.ts";
import { usePlan } from "../../store/plan.ts";

const CLEANING = [
  { id: "light", label: "Light", hint: "Throws away only the worst pages." },
  { id: "standard", label: "Standard", hint: "Removes garbled and low-quality pages." },
  { id: "thorough", label: "Thorough", hint: "Keeps only clean, well-written text. Less to read." },
] as const;

/** Examples of what an option removes, from the datasets in the mix (at most three). */
function examples(
  catalog: PublicCatalog,
  plan: TrainingPlan,
  pick: (d: PublicCatalog["datasets"][number]) => string[],
) {
  return catalog.datasets
    .filter((d) => plan.mix[d.id] > 0)
    .flatMap(pick)
    .slice(0, 3);
}

function Removed({ items }: { items: string[] }) {
  if (items.length === 0) return <p className="text-xs text-muted">Nothing in your mix.</p>;
  return (
    <ul className="space-y-2" aria-label="What this removes">
      {items.map((t) => (
        <li
          key={t}
          className="player-text rounded-lg border border-border bg-surface-2 p-3 text-xs line-through decoration-danger/60"
        >
          {t}
        </li>
      ))}
    </ul>
  );
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
  children,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  children?: ReactNode;
}) {
  return (
    <Card>
      <label className="flex items-start gap-3">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="mt-1 size-4 accent-[var(--brand)]"
        />
        <span>
          <span className="block font-semibold">{label}</span>
          <span className="block text-sm text-muted">{hint}</span>
        </span>
      </label>
      {children && <div className="mt-3">{children}</div>}
    </Card>
  );
}

/** How the reading is cleaned before the AI sees it. */
export function PrepareStep({
  plan,
  catalog,
  balance,
}: {
  plan: TrainingPlan;
  catalog: PublicCatalog;
  balance: Balance;
}) {
  const update = usePlan((s) => s.update);
  const key = prepKey(plan.cleaning, plan.removeDuplicates, plan.factCheck);
  const pool = catalog.datasets
    .filter((d) => plan.mix[d.id] > 0)
    .reduce((sum, d) => sum + (d.pools[key]?.tokens ?? 0), 0);

  return (
    <div className="space-y-4">
      <Card title="Cleaning" hint="How picky to be about what your AI reads.">
        <fieldset
          className="m-0 mb-3 inline-flex rounded-lg border border-border p-1"
          aria-label="Cleaning"
        >
          {CLEANING.map((c) => (
            <button
              key={c.id}
              type="button"
              aria-pressed={plan.cleaning === c.id}
              onClick={() => update({ cleaning: c.id })}
              className={`rounded-md px-4 py-1.5 text-sm ${plan.cleaning === c.id ? "bg-brand text-on-brand" : "text-muted hover:text-text"}`}
            >
              {c.label}
            </button>
          ))}
        </fieldset>
        <p className="mb-3 text-sm text-muted">
          {CLEANING.find((c) => c.id === plan.cleaning)?.hint}
        </p>
        <Removed
          items={examples(catalog, plan, (d) => d.removedExamples.cleaning[plan.cleaning])}
        />
      </Card>
      <Toggle
        label="Remove duplicates"
        hint="Reads each text once, instead of memorizing what appears many times."
        checked={plan.removeDuplicates}
        onChange={(removeDuplicates) => update({ removeDuplicates })}
      >
        {plan.removeDuplicates && (
          <Removed items={examples(catalog, plan, (d) => d.removedExamples.duplicates)} />
        )}
      </Toggle>
      <Toggle
        label={`Fact-checking (${formatMoney(balance.factCheckCost)})`}
        hint="Pays checkers to take out statements that are false."
        checked={plan.factCheck}
        onChange={(factCheck) => update({ factCheck })}
      >
        {plan.factCheck && (
          <Removed items={examples(catalog, plan, (d) => d.removedExamples.factCheck)} />
        )}
      </Toggle>
      <Toggle
        label="Balance the topics"
        hint="Reads about every topic evenly, instead of mostly what the data has most of."
        checked={plan.variety === "balanced"}
        onChange={(on) => update({ variety: on ? "balanced" : "natural" })}
      />
      <p data-testid="pool" className="text-sm">
        After preparing: <span className="font-semibold text-brand">{words(pool)}</span> left to
        learn from.
      </p>
    </div>
  );
}
