import type { NewsView, PublicCompany } from "@airace/protocol";
import { Logo } from "./Logo.tsx";

function moodColor(sentiment: number): string {
  if (sentiment > 0.2) return "var(--good)";
  if (sentiment < -0.2) return "var(--danger)";
  return "var(--muted)";
}

/**
 * A news story: the game's own headline about something that really happened.
 * With `onOpen` the whole card opens what it is about.
 */
export function NewsCard({
  news,
  onOpen,
  company,
}: {
  news: NewsView;
  onOpen?: (el: HTMLElement) => void;
  /** A rival the story is about: its logo shows beside it. */
  company?: PublicCompany;
}) {
  const color = moodColor(news.sentiment);
  // Trash-talk is the game's own words for a rival's founder, never an AI's (R1).
  const commentary = news.topic === "trashTalk";
  return (
    <article
      className={`relative rounded-xl border border-border bg-surface/40 px-4 py-3 ${onOpen ? "lift" : ""}`}
      style={{ borderLeft: `3px solid ${color}` }}
    >
      <p
        className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider"
        style={{ color }}
      >
        {company && <Logo logo={company.logo} palette={company.palette} size={16} />}
        News · day {news.day}
        {commentary && (
          <span className="rounded-full bg-surface-2 px-2 py-0.5 normal-case tracking-normal text-muted">
            Game commentary
          </span>
        )}
      </p>
      <p className={`player-text mt-1 font-display font-semibold ${commentary ? "italic" : ""}`}>
        {news.headline}
      </p>
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
