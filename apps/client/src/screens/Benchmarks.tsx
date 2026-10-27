import type { VersionView, WorldView } from "@airace/protocol";
import { useState } from "react";
import { Card } from "../components/Card.tsx";
import { INPUT } from "../components/Field.tsx";

/** The eight bench-v1 categories in plain words (R7). */
export const CATEGORIES: { id: string; name: string; what: string }[] = [
  { id: "language", name: "Language", what: "Writes sentences that read like real text." },
  { id: "reasoning", name: "Reasoning", what: "Works through simple problems step by step." },
  { id: "pattern", name: "Patterns", what: "Spots a pattern and continues it." },
  { id: "knowledge", name: "Knowledge", what: "Knows facts about the world." },
  { id: "coding", name: "Coding", what: "Writes small programs that work." },
  { id: "creativity", name: "Creativity", what: "Tells new stories instead of copying old ones." },
  { id: "consistency", name: "Consistency", what: "Gives the same answer when asked again." },
  { id: "instruction", name: "Following instructions", what: "Does what it is asked." },
];

function Change({
  now,
  before,
}: {
  now: number | null | undefined;
  before: number | null | undefined;
}) {
  if (now == null || before == null) return null;
  const d = Math.round(now - before);
  if (d === 0) return <span className="text-xs text-muted">no change</span>;
  return (
    <span className={`text-xs font-semibold ${d > 0 ? "text-brand" : "text-danger"}`}>
      {d > 0 ? `▲ ${d}` : `▼ ${-d}`}
    </span>
  );
}

/** A version's measured scores, compared with the version it grew from. Measurements only (R5). */
export function Benchmarks({ view }: { view: WorldView }) {
  const versions = view.me.company?.versions ?? [];
  const [chosen, setChosen] = useState<string | null>(null);
  const version = versions.find((v) => v.id === chosen) ?? versions.at(-1);
  if (!version)
    return <p className="text-muted">Scores appear once your first AI finishes training.</p>;
  const parent: VersionView | undefined = versions.find((v) => v.id === version.parentId);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-4">
        <label className="text-sm">
          <span className="mb-1 block font-semibold">Version</span>
          <select className={INPUT} value={version.id} onChange={(e) => setChosen(e.target.value)}>
            {[...versions].reverse().map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
        </label>
        <p className="text-sm text-muted">
          Scores run from 0 to 100; 0 means no better than guessing.
        </p>
      </div>
      {version.scores === null ? (
        <Card>
          <p className="text-muted">
            Scoring {version.label}… This takes a little while after training ends.
          </p>
        </Card>
      ) : (
        <>
          <Card>
            <div className="flex flex-wrap items-baseline gap-4">
              <p className="text-4xl font-bold text-brand" data-testid="overall">
                {version.overall === null ? "–" : Math.round(version.overall)}
              </p>
              <p className="text-sm text-muted">overall</p>
              <Change now={version.overall} before={parent?.overall} />
              {version.personalityWords.length > 0 && (
                <p className="ml-auto text-sm">
                  Personality:{" "}
                  <span className="font-semibold">{version.personalityWords.join(", ")}</span>
                </p>
              )}
            </div>
            {parent && (
              <p className="mt-1 text-xs text-muted">
                Compared with {parent.label}, the AI it grew from.
              </p>
            )}
          </Card>
          <div className="grid gap-3 md:grid-cols-2">
            {CATEGORIES.map((c) => {
              const score = version.scores?.[c.id];
              return (
                <Card key={c.id} data-testid={`score-${c.id}`}>
                  <div className="flex items-baseline justify-between">
                    <h3 className="font-semibold">{c.name}</h3>
                    <span className="flex items-baseline gap-2">
                      <Change now={score} before={parent?.scores?.[c.id]} />
                      <span className="text-lg font-bold">
                        {score == null ? "Not measured" : Math.round(score)}
                      </span>
                    </span>
                  </div>
                  <p className="text-xs text-muted">{c.what}</p>
                  {score != null && (
                    <progress
                      className="mt-2 h-2 w-full accent-[var(--brand)]"
                      value={score}
                      max={100}
                      aria-label={c.name}
                    />
                  )}
                </Card>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
