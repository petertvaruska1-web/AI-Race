import type { FeedItemView, PublicCompany } from "@airace/protocol";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { useWindows } from "../store/windows.ts";
import { NewsCard } from "./NewsCard.tsx";
import { PostCard } from "./PostCard.tsx";
import { Toggle } from "./Toggle.tsx";

/**
 * Posts and news, newest first; new ones slide in at the top. A post opens its
 * window; news about the player's own company opens the customers window.
 */
export function Feed({
  items: all,
  companyId = null,
  companies = [],
}: {
  items: FeedItemView[];
  companyId?: string | null;
  /** With rivals among them, the feed can show only stories about you, and rivals' logos. */
  companies?: PublicCompany[];
}) {
  const open = useWindows((s) => s.open);
  const [onlyYou, setOnlyYou] = useState(false);
  const rivals = companies.some((c) => c.rival !== null);
  const items = onlyYou ? all.filter((i) => i.companyId === companyId) : all;
  const filter = rivals && companyId !== null && (
    <Toggle
      legend="Show"
      options={[
        { label: "Everything", value: "all" },
        { label: "About you", value: "you" },
      ]}
      value={onlyYou ? "you" : "all"}
      onChange={(v) => setOnlyYou(v === "you")}
    />
  );
  if (items.length === 0) {
    return (
      <>
        {filter}
        <p className="text-sm text-muted">
          Nothing yet. Launch your AI, and customers will start talking about it here.
        </p>
      </>
    );
  }
  return (
    <>
      {filter}
      <ol className="space-y-3" aria-label="News and posts">
        <AnimatePresence initial={false}>
          {[...items].reverse().map((item) => (
            <motion.li
              key={item.id}
              layout
              initial={{ opacity: 0, y: -18, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.97 }}
              transition={{ type: "spring", stiffness: 320, damping: 30 }}
            >
              {item.kind === "post" ? (
                <PostCard
                  post={item}
                  onOpen={(el) => open({ kind: "post", postId: item.id }, el)}
                />
              ) : (
                <NewsCard
                  news={item}
                  company={
                    item.companyId !== companyId
                      ? companies.find((c) => c.id === item.companyId && c.rival)
                      : undefined
                  }
                  onOpen={
                    companyId !== null && item.companyId === companyId
                      ? (el) => open({ kind: "customers" }, el)
                      : undefined
                  }
                />
              )}
            </motion.li>
          ))}
        </AnimatePresence>
      </ol>
    </>
  );
}
