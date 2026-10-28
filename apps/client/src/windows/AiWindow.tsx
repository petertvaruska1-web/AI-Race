import { motion } from "motion/react";
import { Button } from "../components/Button.tsx";
import { dayLabel, words } from "../format.ts";
import { Orb } from "../orb/Orb.tsx";
import { go } from "../route.ts";
import { CATEGORIES } from "../screens/Benchmarks.tsx";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, Section, WindowFrame, type WindowProps } from "./WindowFrame.tsx";

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

/** The measured style traits in plain words, each scaled to 0..1 for its bar. */
const TRAITS: { id: string; name: string; scale: (v: number) => number }[] = [
  { id: "verbosity", name: "Says a lot", scale: (v) => Math.max(0, v) / (Math.max(0, v) + 25) },
  { id: "inventiveness", name: "Inventive", scale: clamp01 },
  { id: "steadiness", name: "Steady", scale: clamp01 },
  { id: "boldness", name: "Bold", scale: clamp01 },
  { id: "warmth", name: "Warm", scale: (v) => Math.max(0, v) / (Math.max(0, v) + 0.4) },
  { id: "slip_rate", name: "Slips up", scale: clamp01 },
];

function Bar({ share, color, delay }: { share: number; color: string; delay: number }) {
  return (
    <div className="mt-1 h-2 rounded-full bg-surface-2">
      <motion.div
        className="h-2 rounded-full"
        style={{ background: color }}
        initial={{ width: 0 }}
        animate={{ width: `${100 * clamp01(share)}%` }}
        transition={{ duration: 0.8, ease: "easeOut", delay }}
      />
    </div>
  );
}

/** One AI: its orb, scores, style, where it grew from and where it is on sale. */
export function AiWindow({ entry, view, versionId }: WindowProps & { versionId: string }) {
  const open = useWindows((s) => s.open);
  const company = view.me.company;
  const pub = view.companies.find((c) => c.id === company?.id);
  const v = company?.versions.find((x) => x.id === versionId);
  if (!company || !pub || !v) return <GoneWindow entry={entry} />;
  const parent = company.versions.find((x) => x.id === v.parentId);
  const products = company.products.filter((p) => p.versionId === v.id && p.status !== "retired");
  const to = (tab: "lab" | "play") => go({ screen: "world", worldId: view.world.id, tab });
  return (
    <WindowFrame
      entry={entry}
      title={`Your AI ${v.label}`}
      subtitle={
        v.personalityWords.length > 0
          ? v.personalityWords.join(", ")
          : v.overall === null
            ? "Being scored…"
            : undefined
      }
      wide
    >
      <div className="grid items-center gap-6 sm:grid-cols-[auto_1fr]">
        <div className="mx-auto">
          <Orb
            size={200}
            label={`Your AI ${v.label}`}
            input={{
              overall: v.overall,
              scores: v.scores,
              traits: v.traits,
              trainedTokens: v.trainedTokens,
              palette: pub.palette,
              training: null,
            }}
          />
        </div>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-muted">Overall</p>
            <p className="font-display text-4xl font-bold text-brand">
              {v.overall === null ? "…" : Math.round(v.overall)}
            </p>
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-muted">Has read</p>
            <p className="mt-1 font-semibold">{words(v.trainedTokens)}</p>
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-muted">Born</p>
            <p className="mt-1 font-semibold">{dayLabel(v.createdDay)}</p>
          </div>
        </div>
      </div>

      <Section title="Scores">
        {v.scores ? (
          <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
            {CATEGORIES.map((c, i) => {
              const score = v.scores?.[c.id];
              if (score === null || score === undefined) return null;
              return (
                // biome-ignore lint/a11y/useSemanticElements: a styled, animated bar; a native meter can't be drawn alike in every browser
                <div
                  key={c.id}
                  role="meter"
                  aria-label={c.name}
                  aria-valuenow={Math.round(score)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  title={c.what}
                >
                  <p className="flex justify-between text-sm">
                    <span>{c.name}</span>
                    <span className="tabular font-semibold">{Math.round(score)}</span>
                  </p>
                  <Bar share={score / 100} color="var(--brand)" delay={0.05 * i} />
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-sm text-muted">Its scores arrive once it has been tested.</p>
        )}
      </Section>

      {v.traits && (
        <Section title="Its style">
          <div className="grid gap-x-6 gap-y-3 sm:grid-cols-3">
            {TRAITS.filter((t) => typeof v.traits?.[t.id] === "number").map((t, i) => (
              <div key={t.id}>
                <p className="text-sm">{t.name}</p>
                <Bar
                  share={t.scale(v.traits?.[t.id] ?? 0)}
                  color="var(--brand-2)"
                  delay={0.05 * i}
                />
              </div>
            ))}
          </div>
        </Section>
      )}

      {(parent || products.length > 0) && (
        <div className="grid gap-x-6 sm:grid-cols-2">
          {parent && (
            <Section title="Grew from">
              <button
                type="button"
                onClick={(e) => open({ kind: "ai", versionId: parent.id }, e.currentTarget)}
                className="row-btn"
              >
                <span>Your AI {parent.label}</span>
                <span aria-hidden="true" className="ml-auto text-muted">
                  ›
                </span>
              </button>
            </Section>
          )}
          {products.length > 0 && (
            <Section title="On sale as">
              <ul className="space-y-2">
                {products.map((p) => (
                  <li key={p.id}>
                    <button
                      type="button"
                      onClick={(e) => open({ kind: "product", productId: p.id }, e.currentTarget)}
                      className="row-btn"
                    >
                      <span className="player-text font-semibold">{p.name}</span>
                      <span aria-hidden="true" className="ml-auto text-muted">
                        ›
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </Section>
          )}
        </div>
      )}

      <div className="mt-6 flex flex-wrap gap-3">
        <Button onClick={() => to("play")}>Chat with it</Button>
        <Button variant="secondary" onClick={() => to("lab")}>
          Train it further
        </Button>
      </div>
    </WindowFrame>
  );
}
