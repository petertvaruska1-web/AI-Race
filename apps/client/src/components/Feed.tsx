import type { FeedItemView } from "@airace/protocol";
import { NewsCard } from "./NewsCard.tsx";
import { PostCard } from "./PostCard.tsx";

/** Posts and news, newest first. */
export function Feed({ items }: { items: FeedItemView[] }) {
  if (items.length === 0) {
    return (
      <p className="text-sm text-muted">
        Nothing yet. Launch your AI, and customers will start talking about it here.
      </p>
    );
  }
  return (
    <ol className="space-y-3" aria-label="News and posts">
      {[...items].reverse().map((item) => (
        <li key={item.id}>
          {item.kind === "post" ? <PostCard post={item} /> : <NewsCard news={item} />}
        </li>
      ))}
    </ol>
  );
}
