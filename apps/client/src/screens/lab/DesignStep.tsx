import { TEXT_LIMITS, type TrainingPlan, type VersionView } from "@airace/protocol";
import { type Balance, paramCount, previewPlan } from "@airace/sim";
import { KeyRow } from "../../bench/KeyRow.tsx";
import { Module } from "../../bench/Module.tsx";
import { Slider } from "../../bench/Slider.tsx";
import { capitalize, formatMoney, gameDays, percent, words } from "../../format.ts";
import { TrainingExtras } from "../../spending/Spending.tsx";
import { usePlan } from "../../store/plan.ts";

/**
 * The AI's design (design §10, the AI Lab): what it grows from, how big its brain is, how much it
 * keeps in mind, how long and how it learns, as lit keys and sliders with their plain
 * explanations; the paid extras and the test question beside them.
 */
export function DesignStep({
  plan,
  balance,
  versions,
  lent = 0,
  loved = 0,
  maxParams = null,
  limitNote = null,
}: {
  plan: TrainingPlan;
  balance: Balance;
  versions: VersionView[];
  /** Conversations customers loved, for "Learn from customers". */
  loved?: number;
  /** Data-center compute lent to training (PFLOP a day). */
  lent?: number;
  /** The biggest brain the cards can train, in parameters (null: no limit known). */
  maxParams?: number | null;
  /** Why a bigger brain is off. */
  limitNote?: string | null;
}) {
  const update = usePlan((s) => s.update);
  const parent = versions.find((v) => v.id === plan.parentVersionId) ?? null;
  const [, maxPrompt] = TEXT_LIMITS.probePrompt;
  const shortest = Math.min(...balance.attentionSpans);
  const longest = Math.max(...balance.attentionSpans);
  return (
    <div className="lab-design">
      <div className="lab-design__col">
        {versions.length > 0 && (
          <Module
            label="Grow from"
            hint="A new version keeps everything the earlier one learned and builds on it."
          >
            <select
              className="display-input lab-select"
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
          </Module>
        )}
        <Module
          label="Brain size"
          weight="heavy"
          hint="A bigger brain can learn more, but costs more and takes longer to train, and costs more to run for every customer."
        >
          <KeyRow
            legend="Brain size"
            className="keyrow--cards"
            options={balance.brainSizes.map((b) => {
              const shrinks =
                parent !== null &&
                (b.nLayer < parent.shape.nLayer || b.dModel < parent.shape.dModel);
              const p = previewPlan(
                balance,
                parent?.shape ?? null,
                { ...plan, brainSize: b.id },
                lent,
              );
              const tooBig = maxParams !== null && paramCount(b.nLayer, b.dModel) > maxParams;
              return {
                label: capitalize(b.id),
                value: b.id,
                disabled: shrinks || tooBig,
                ...(p.ok
                  ? {
                      detail: `${formatMoney(p.preview.cost)} · ${gameDays(p.preview.durationDays)}`,
                    }
                  : {}),
                ...(shrinks
                  ? { note: "Your AI can only grow" }
                  : tooBig && limitNote
                    ? { note: limitNote }
                    : {}),
              };
            })}
            value={plan.brainSize}
            onChange={(brainSize) => update({ brainSize })}
          />
        </Module>
        <Module
          label="Attention span"
          hint="How much text it keeps in mind at once while reading and answering."
        >
          <KeyRow
            legend="Attention span"
            className="keyrow--cards"
            options={balance.attentionSpans.map((span) => ({
              label: `${span} words`,
              value: String(span),
              detail: span === shortest ? "Short" : span === longest ? "Long" : "Medium",
            }))}
            value={String(plan.attentionSpan)}
            onChange={(span) => update({ attentionSpan: Number(span) })}
          />
        </Module>
      </div>
      <div className="lab-design__col">
        <Module
          label="Training length"
          hint="How much it reads in this run. Longer costs more and takes more game days."
        >
          <KeyRow
            legend="Training length"
            className="keyrow--cards"
            options={balance.lengths.map((l) => ({
              label: capitalize(l.id),
              value: l.id,
              detail: words(l.tokens),
            }))}
            value={plan.length}
            onChange={(length) => update({ length })}
          />
        </Module>
        <Module
          label="Learning style"
          hint="Bold learns faster but can stumble; careful is slow and steady."
        >
          <div className="lab-ends">
            <span>Careful</span>
            <Slider
              label="Learning style"
              value={plan.learningStyle}
              min={0}
              max={1}
              step={0.05}
              onChange={(learningStyle) => update({ learningStyle })}
              showLabel={false}
            />
            <span>Bold</span>
          </div>
        </Module>
        <Module
          label={`Review what it knows: ${percent(plan.review)}`}
          hint={
            parent
              ? "Part of the reading repeats what it read before, so it doesn't forget old skills."
              : "Needs an earlier AI to review."
          }
        >
          <Slider
            label="Review what it knows"
            value={plan.review}
            min={0}
            max={0.5}
            step={0.05}
            disabled={!parent}
            onChange={(review) => update({ review })}
            showLabel={false}
          />
        </Module>
      </div>
      <div className="lab-design__col">
        <TrainingExtras plan={plan} loved={loved} />
        <Module
          label="Your test question (optional)"
          hint="We ask your AI this while it trains, so you can watch its answer improve."
        >
          <input
            className="display-input"
            aria-label="Your test question (optional)"
            maxLength={maxPrompt}
            value={plan.probePrompt}
            placeholder="What is the sun?"
            onChange={(e) => update({ probePrompt: e.target.value })}
          />
        </Module>
      </div>
    </div>
  );
}
