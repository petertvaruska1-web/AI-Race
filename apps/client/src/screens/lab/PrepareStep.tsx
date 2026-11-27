import { type PublicCatalog, prepKey, type TrainingPlan } from "@airace/protocol";
import type { Balance } from "@airace/sim";
import type { ReactNode } from "react";
import { KeyRow } from "../../bench/KeyRow.tsx";
import { Module } from "../../bench/Module.tsx";
import { ToggleSwitch } from "../../bench/ToggleSwitch.tsx";
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
  if (items.length === 0) return <p className="lab-note">Nothing in your mix.</p>;
  return (
    <ul className="display lab-removed" aria-label="What this removes">
      {items.map((t) => (
        <li key={t} className="player-text lab-removed__item">
          {t}
        </li>
      ))}
    </ul>
  );
}

/** One option of the preparing: a switch with its lamp, and what it removes while on. */
function Option({
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
    <div className="lab-option">
      <ToggleSwitch label={label} hint={hint} checked={checked} onChange={onChange} />
      {children}
    </div>
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
    <div className="lab-prepare">
      <Module
        label="Cleaning"
        weight="heavy"
        hint="How picky to be about what your AI reads."
        className="lab-prepare__cleaning"
      >
        <KeyRow
          legend="Cleaning"
          options={CLEANING.map((c) => ({ label: c.label, value: c.id }))}
          value={plan.cleaning}
          onChange={(cleaning) => update({ cleaning })}
          className="lab-prepare__keys"
        />
        <p className="lab-note">{CLEANING.find((c) => c.id === plan.cleaning)?.hint}</p>
        <Removed
          items={examples(catalog, plan, (d) => d.removedExamples.cleaning[plan.cleaning])}
        />
        <p data-testid="pool" className="display lab-prepare__pool">
          After preparing: <span className="dotface dotface--sm">{words(pool)}</span> left to learn
          from.
        </p>
      </Module>
      <Module label="Before your AI reads it" className="lab-prepare__options">
        <Option
          label="Remove duplicates"
          hint="Reads each text once, instead of memorizing what appears many times."
          checked={plan.removeDuplicates}
          onChange={(removeDuplicates) => update({ removeDuplicates })}
        >
          {plan.removeDuplicates && (
            <Removed items={examples(catalog, plan, (d) => d.removedExamples.duplicates)} />
          )}
        </Option>
        <Option
          label={`Fact-checking (${formatMoney(balance.factCheckCost)})`}
          hint="Pays checkers to take out statements that are false."
          checked={plan.factCheck}
          onChange={(factCheck) => update({ factCheck })}
        >
          {plan.factCheck && (
            <Removed items={examples(catalog, plan, (d) => d.removedExamples.factCheck)} />
          )}
        </Option>
        <Option
          label={`Fact-check plus ($${balance.economy.expertCheckPerThousand} per thousand words)`}
          hint="Experts keep only the cleanest text, without duplicates or false statements."
          checked={plan.cleaning === "expert"}
          onChange={(on) => update({ cleaning: on ? "expert" : "thorough" })}
        >
          {plan.cleaning === "expert" && (
            <Removed items={examples(catalog, plan, (d) => d.removedExamples.cleaning.expert)} />
          )}
        </Option>
        <Option
          label="Balance the topics"
          hint="Reads about every topic evenly, instead of mostly what the data has most of."
          checked={plan.variety === "balanced"}
          onChange={(on) => update({ variety: on ? "balanced" : "natural" })}
        />
      </Module>
    </div>
  );
}
