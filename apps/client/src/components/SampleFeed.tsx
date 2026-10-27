export type SampleEvent = { step: number; prompt: string; text: string };

/** What the AI says while it trains, newest first. Its words are shown exactly, as text (R1). */
export function SampleFeed({
  samples,
  totalSteps,
}: {
  samples: SampleEvent[];
  totalSteps: number;
}) {
  if (samples.length === 0) {
    return <p className="text-sm text-muted">Its first words appear here as it learns.</p>;
  }
  return (
    <ol className="space-y-3" data-testid="sample-feed">
      {[...samples].reverse().map((s) => (
        <li
          key={`${s.step}:${s.prompt}`}
          className="rounded-xl border border-border bg-surface-2 p-3"
        >
          <p className="mb-1 text-xs text-muted">
            {totalSteps > 0
              ? `${Math.round((100 * s.step) / totalSteps)}% through`
              : `Step ${s.step}`}{" "}
            · asked: <span className="player-text">{s.prompt}</span>
          </p>
          <p className="player-text text-sm">{s.text}</p>
        </li>
      ))}
    </ol>
  );
}
