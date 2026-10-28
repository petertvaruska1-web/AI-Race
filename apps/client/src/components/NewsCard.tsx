import type { NewsView } from "@airace/protocol";

function moodColor(sentiment: number): string {
  if (sentiment > 0.2) return "var(--good)";
  if (sentiment < -0.2) return "var(--danger)";
  return "var(--muted)";
}

/** A news story: the game's own headline about something that really happened. */
export function NewsCard({ news }: { news: NewsView }) {
  const color = moodColor(news.sentiment);
  return (
    <article
      className="rounded-xl border border-border bg-surface/40 px-4 py-3"
      style={{ borderLeft: `3px solid ${color}` }}
    >
      <p className="text-[11px] font-semibold uppercase tracking-wider" style={{ color }}>
        News · day {news.day}
      </p>
      <p className="player-text mt-1 font-display font-semibold">{news.headline}</p>
    </article>
  );
}
