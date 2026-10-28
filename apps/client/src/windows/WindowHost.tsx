import type { WorldView } from "@airace/protocol";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef } from "react";
import { type OpenWindow, useWindows } from "../store/windows.ts";
import { AiWindow } from "./AiWindow.tsx";
import { CampaignWindow } from "./CampaignWindow.tsx";
import { CustomersWindow } from "./CustomersWindow.tsx";
import { ComputersWindow, DataCenterWindow } from "./DataCenterWindow.tsx";
import { GroupWindow } from "./GroupWindow.tsx";
import { CategoryWindow, MoneyWindow, MonthWindow } from "./MoneyWindows.tsx";
import { PostWindow } from "./PostWindow.tsx";
import { ProductWindow } from "./ProductWindow.tsx";
import { RegionWindow } from "./RegionWindow.tsx";
import { ReportWindow } from "./ReportWindow.tsx";

function WindowContent({ entry, view }: { entry: OpenWindow; view: WorldView }) {
  const s = entry.spec;
  switch (s.kind) {
    case "customers":
      return <CustomersWindow entry={entry} view={view} />;
    case "money":
      return <MoneyWindow entry={entry} view={view} />;
    case "group":
      return <GroupWindow entry={entry} view={view} group={s.group} />;
    case "product":
      return <ProductWindow entry={entry} view={view} productId={s.productId} />;
    case "ai":
      return <AiWindow entry={entry} view={view} versionId={s.versionId} />;
    case "month":
      return <MonthWindow entry={entry} view={view} month={s.month} />;
    case "income":
      return <CategoryWindow entry={entry} view={view} side="income" category={s.category} />;
    case "cost":
      return <CategoryWindow entry={entry} view={view} side="costs" category={s.category} />;
    case "post":
      return <PostWindow entry={entry} view={view} postId={s.postId} />;
    case "region":
      return <RegionWindow entry={entry} view={view} region={s.region} />;
    case "campaign":
      return <CampaignWindow entry={entry} view={view} region={s.region} />;
    case "dataCenter":
      return <DataCenterWindow entry={entry} view={view} />;
    case "computers":
      return <ComputersWindow entry={entry} view={view} />;
    case "report":
      return <ReportWindow entry={entry} view={view} month={s.month} />;
  }
}

/**
 * Shows the top window of the stack over a dimmed screen (immersion design).
 * Escape goes back one; the backdrop closes all; the page's opener gets the
 * focus back when the last one closes. Windows show the live view.
 */
export function WindowHost({ view }: { view: WorldView }) {
  const stack = useWindows((s) => s.stack);
  const back = useWindows((s) => s.back);
  const closeAll = useWindows((s) => s.closeAll);
  const top = stack.at(-1);
  const opener = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (stack[0]) {
      opener.current = stack[0].opener;
      return;
    }
    if (opener.current?.isConnected) opener.current.focus();
    opener.current = null;
  }, [stack]);

  useEffect(() => {
    if (!top) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      back();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [top, back]);

  return (
    <AnimatePresence>
      {top && (
        <motion.div
          key="windows"
          className="fixed inset-0 z-40 flex items-center justify-center p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          // Clicks pass through while it fades away.
          exit={{ opacity: 0, pointerEvents: "none" }}
          transition={{ duration: 0.18 }}
        >
          <button
            type="button"
            aria-label="Close all windows"
            tabIndex={-1}
            onClick={closeAll}
            className="absolute inset-0 cursor-default bg-bg/70 backdrop-blur-sm"
          />
          <AnimatePresence mode="wait">
            <WindowContent key={top.id} entry={top} view={view} />
          </AnimatePresence>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
