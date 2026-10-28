import { TEXT_LIMITS, type TrainingPlan, type VersionView } from "@airace/protocol";
import { type Balance, previewPlan } from "@airace/sim";
import { Field, INPUT } from "../../components/Field.tsx";
import { Panel } from "../../components/Panel.tsx";
import { capitalize, formatMoney, gameDays, percent, words } from "../../format.ts";
import { usePlan } from "../../store/plan.ts";

function Choice({
  selected,
  disabled,
  onClick,
  title,
  detail,
  note,
}: {
  selected: boolean;
  disabled?: boolean;
  onClick: () => void;
  title: string;
  detail: string;
  note?: string;
}) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      disabled={disabled}
      onClick={onClick}
      className={`rounded-xl border p-3 text-left disabled:cursor-not-allowed disabled:opacity-40 ${selected ? "border-brand bg-brand-soft" : "border-border hover:border-brand"}`}
    >
      <span className="block font-semibold">{title}</span>
      <span className="block text-xs text-muted">{detail}</span>
      {note && <span className="mt-1 block text-xs text-danger">{note}</span>}
    </button>
  );
}

/** The AI's design: how big its brain is, how much it reads at once, how long and how it learns. */
export function DesignStep({
  plan,
  balance,
  versions,
}: {
  plan: TrainingPlan;
  balance: Balance;
  versions: VersionView[];
}) {
  const update = usePlan((s) => s.update);
  const parent = versions.find((v) => v.id === plan.parentVersionId) ?? null;
  const [, maxPrompt] = TEXT_LIMITS.probePrompt;

  return (
    <div className="space-y-4">
      {versions.length > 0 && (
        <Panel
          title="Grow from"
          hint="A new version keeps everything the earlier one learned and builds on it."
        >
          <select
            className={INPUT}
            aria-label="Grow from"
            value={plan.parentVersionId ?? ""}
            onChange={(e) =>
              update({
                parentVersionId: e.target.value || null,
                ...(e.target.value ? {} : { review: 0 }),
              })
            }
          >
            {versions.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
            <option value="">Start a brand-new AI</option>
          </select>
        </Panel>
      )}

      <Panel
        title="Brain size"
        hint="A bigger brain can learn more, but costs more and takes longer to train."
      >
        <div className="grid gap-3 sm:grid-cols-3">
          {balance.brainSizes.map((b) => {
            const shrinks =
              parent !== null && (b.nLayer < parent.shape.nLayer || b.dModel < parent.shape.dModel);
            const p = previewPlan(balance, parent?.shape ?? null, { ...plan, brainSize: b.id });
            return (
              <Choice
                key={b.id}
                selected={plan.brainSize === b.id}
                disabled={shrinks}
                onClick={() => update({ brainSize: b.id })}
                title={capitalize(b.id)}
                detail={
                  p.ok ? `${formatMoney(p.preview.cost)} · ${gameDays(p.preview.durationDays)}` : ""
                }
                {...(shrinks ? { note: "Your AI can only grow" } : {})}
              />
            );
          })}
        </div>
      </Panel>

      <Panel
        title="Attention span"
        hint="How much text it keeps in mind at once while reading and answering."
      >
        <div className="grid gap-3 sm:grid-cols-3">
          {balance.attentionSpans.map((span) => (
            <Choice
              key={span}
              selected={plan.attentionSpan === span}
              onClick={() => update({ attentionSpan: span })}
              title={`${span} words`}
              detail={
                span === Math.min(...balance.attentionSpans)
                  ? "Short"
                  : span === Math.max(...balance.attentionSpans)
                    ? "Long"
                    : "Medium"
              }
            />
          ))}
        </div>
      </Panel>

      <Panel
        title="Training length"
        hint="How much it reads in this run. Longer costs more and takes more game days."
      >
        <div className="grid gap-3 sm:grid-cols-3">
          {balance.lengths.map((l) => (
            <Choice
              key={l.id}
              selected={plan.length === l.id}
              onClick={() => update({ length: l.id })}
              title={capitalize(l.id)}
              detail={words(l.tokens)}
            />
          ))}
        </div>
      </Panel>

      <Panel
        title="Learning style"
        hint="Bold learns faster but can stumble; careful is slow and steady."
      >
        <div className="flex items-center gap-3 text-sm">
          <span className="text-muted">Careful</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            aria-label="Learning style"
            value={plan.learningStyle}
            onChange={(e) => update({ learningStyle: Number(e.target.value) })}
            className="w-full accent-[var(--brand)]"
          />
          <span className="text-muted">Bold</span>
        </div>
      </Panel>

      <Panel
        title={`Review what it knows: ${percent(plan.review)}`}
        hint={
          parent
            ? "Part of the reading repeats what it read before, so it doesn't forget old skills."
            : "Needs an earlier AI to review."
        }
      >
        <input
          type="range"
          min={0}
          max={0.5}
          step={0.05}
          aria-label="Review what it knows"
          disabled={!parent}
          value={plan.review}
          onChange={(e) => update({ review: Number(e.target.value) })}
          className="w-full accent-[var(--brand)] disabled:opacity-40"
        />
      </Panel>

      <Panel>
        <Field
          label="Your test question (optional)"
          hint="We ask your AI this while it trains, so you can watch its answer improve."
        >
          <input
            className={INPUT}
            maxLength={maxPrompt}
            value={plan.probePrompt}
            placeholder="What is the sun?"
            onChange={(e) => update({ probePrompt: e.target.value })}
          />
        </Field>
      </Panel>
    </div>
  );
}
