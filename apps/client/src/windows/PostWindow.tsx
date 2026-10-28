import type { PostView } from "@airace/protocol";
import { Button } from "../components/Button.tsx";
import { PostCard } from "../components/PostCard.tsx";
import { dayLabel } from "../format.ts";
import { GROUPS } from "../groups.ts";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, WindowFrame, type WindowProps } from "./WindowFrame.tsx";

/** One customer post, the reply quoted whole (R1), and who wrote it. */
export function PostWindow({ entry, view, postId }: WindowProps & { postId: string }) {
  const open = useWindows((s) => s.open);
  const post = view.feed.find((f): f is PostView => f.kind === "post" && f.id === postId);
  if (!post) return <GoneWindow entry={entry} />;
  const g = post.author.group;
  const person = view.spotlights.find((s) => s.id === post.author.spotlightId);
  const company = view.me.company;
  const product =
    post.companyId === company?.id
      ? company?.products.find((p) => p.name === post.productName)
      : undefined;
  return (
    <WindowFrame
      entry={entry}
      title={person?.name ?? post.author.handle}
      subtitle={`${GROUPS[g].one} · ${dayLabel(post.day)}`}
    >
      {person && <p className="player-text mb-4 text-sm text-muted">{person.bio}</p>}
      <PostCard post={post} full />
      <div className="mt-5 flex flex-wrap gap-2">
        <Button
          variant="secondary"
          aria-label={`Open ${GROUPS[g].many}`}
          onClick={(e) => open({ kind: "group", group: g }, e.currentTarget)}
        >
          {GROUPS[g].many} ›
        </Button>
        {product && (
          <Button
            variant="secondary"
            aria-label={`Open ${product.name}`}
            onClick={(e) => open({ kind: "product", productId: product.id }, e.currentTarget)}
          >
            <span className="player-text">{product.name}</span> ›
          </Button>
        )}
      </div>
    </WindowFrame>
  );
}
