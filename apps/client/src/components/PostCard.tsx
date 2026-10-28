import type { PostTone, PostView } from "@airace/protocol";
import { useState } from "react";
import { GROUPS } from "../groups.ts";

const FOLD_AT = 280;

const TONES: Record<PostTone, { color: string; words: string }> = {
  praise: { color: "var(--good)", words: "Loved it" },
  joke: { color: "var(--brand-2)", words: "Had a laugh" },
  roast: { color: "var(--danger)", words: "Roasted it" },
};

/**
 * A customer's post: the game's frame around a question and the AI's reply,
 * quoted exactly and shown as text (R1). A long reply folds, but all of it stays;
 * `full` never folds. With `onOpen` the whole card opens the post's window.
 */
export function PostCard({
  post,
  full = false,
  onOpen,
}: {
  post: PostView;
  full?: boolean;
  onOpen?: (el: HTMLElement) => void;
}) {
  const [open, setOpen] = useState(false);
  const tone = TONES[post.tone];
  const long = !full && post.reply.length > FOLD_AT;
  const spotlight = post.author.spotlightId !== null;
  return (
    <article
      className={`relative rounded-xl border border-border bg-surface/60 p-4 ${onOpen ? "lift" : ""}`}
      style={{ borderLeft: `3px solid ${tone.color}` }}
    >
      <header className="flex flex-wrap items-baseline gap-x-2">
        <span className="player-text font-semibold">
          {spotlight ? post.author.name : post.author.handle}
        </span>
        {spotlight && <span className="player-text text-xs text-muted">{post.author.handle}</span>}
        <span className="ml-auto text-xs text-muted">
          {GROUPS[post.author.group].one} · day {post.day}
        </span>
      </header>
      <p className="mt-2 text-xs text-muted">
        asked <span className="player-text font-semibold text-text">{post.productName}</span>:
      </p>
      <p className="player-text text-sm">{post.question}</p>
      <p className="mt-2 text-xs text-muted">and it said:</p>
      <blockquote
        className={`player-text mt-1 rounded-lg bg-surface-2 p-3 text-sm ${long && !open ? "max-h-36 overflow-hidden" : ""}`}
      >
        {post.reply === "" ? (
          <span className="italic text-muted">(nothing at all)</span>
        ) : (
          post.reply
        )}
      </blockquote>
      <footer className="mt-2 flex items-center justify-between text-xs">
        <span style={{ color: tone.color }} className="font-semibold">
          {tone.words}
        </span>
        {long && (
          <button
            type="button"
            className="relative z-10 text-muted hover:text-text"
            onClick={() => setOpen(!open)}
          >
            {open ? "Show less" : "Show more"}
          </button>
        )}
      </footer>
      {onOpen && (
        <button
          type="button"
          aria-label={`Open the post by ${spotlight ? post.author.name : post.author.handle}`}
          className="absolute inset-0 rounded-xl"
          onClick={(e) => onOpen(e.currentTarget)}
        />
      )}
    </article>
  );
}
