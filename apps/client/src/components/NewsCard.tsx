import type { NewsView } from "@airace/protocol";

function moodColor(sentiment: number): string {
  if (sentiment > 0.2) return "var(--good)";
  if (sentiment < -0.2) return "var(--danger)";
  return "var(--muted)";
}

/**
 * A news story: the game's own headline about something that really happened.
 * With `onOpen` the whole card opens what it is about.
 */
export function NewsCard({ news, onOpen }: { news: NewsView; onOpen?: (el: HTMLElement) => void }) {
  const color = moodColor(news.sentiment);
  return (
    <article
      className={`relative rounded-xl border border-border bg-surface/40 px-4 py-3 ${onOpen ? "lift" : ""}`}
      style={{ borderLeft: `3px solid ${color}` }}
    >
      <p className="text-[11px] font-semibold uppercase tracking-wider" style={{ color }}>
        News · day {news.day}
      </p>
      <p className="player-text mt-1 font-display font-semibold">{news.headline}</p>
      {onOpen && (
        <button
          type="button"
          aria-label={`Open the news: ${news.headline}`}
          className="absolute inset-0 rounded-xl"
          onClick={(e) => onOpen(e.currentTarget)}
        />
      )}
    </article>
  );
}
