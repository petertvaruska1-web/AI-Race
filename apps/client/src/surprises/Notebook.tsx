import type { WorldView } from "@airace/protocol";
import { useEffect, useState } from "react";
import { Button } from "../components/Button.tsx";
import { INPUT } from "../components/Field.tsx";
import { Panel } from "../components/Panel.tsx";
import { formatCount, percent } from "../format.ts";
import { usePlan } from "../store/plan.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { WindowFrame, type WindowProps } from "../windows/WindowFrame.tsx";

export const MAX_NOTEBOOK_WORDS = 5_000;
/** Above this share a small notebook gets learned by heart. */
const BIG_SHARE = 0.2;
export const PARROT_WARNING =
  "Your notebook is small: a big share makes the AI repeat it word for word.";

const wordsIn = (text: string) => text.split(/\s+/).filter(Boolean).length;

/** The ninth card of the Data step: the player's own notebook. */
export function NotebookCard({ view, total }: { view: WorldView; total: number }) {
  const open = useWindows((s) => s.open);
  const plan = usePlan((s) => s.plan);
  const setWeight = usePlan((s) => s.setWeight);
  const words = view.me.company?.notebook?.words ?? 0;
  const weight = plan?.mix.notebook ?? 0;
  const share = total > 0 ? weight / total : 0;
  return (
    <Panel data-testid="dataset-notebook">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h3 className="text-lg font-semibold">Your notebook</h3>
        <span className="text-sm font-semibold text-brand">{percent(share)}</span>
      </div>
      <p className="text-sm text-muted">
        Write your own text for your AI to read: facts, stories, the way you want it to talk. Only
        your company's AI reads it.
      </p>
      <p className="mt-1 text-xs text-muted" data-testid="notebook-words">
        {`${formatCount(words)} / ${formatCount(MAX_NOTEBOOK_WORDS)} words`}
      </p>
      <input
        type="range"
        min={0}
        max={100}
        step={1}
        value={weight}
        disabled={words === 0}
        aria-label="Your notebook share"
        onChange={(e) => setWeight("notebook", Number(e.target.value))}
        className="mt-3 w-full accent-[var(--brand)]"
      />
      {words === 0 && <p className="text-xs text-muted">Write something first to train on it.</p>}
      {share > BIG_SHARE && words > 0 && (
        <p className="mt-1 text-xs text-danger">{PARROT_WARNING}</p>
      )}
      <Button
        variant="secondary"
        className="mt-3"
        onClick={(e) => open({ kind: "notebook" }, e.currentTarget)}
      >
        Write
      </Button>
    </Panel>
  );
}

/** Writing the notebook: a text area with a live word count, saved on the server. */
export function NotebookWindow({ entry, view }: WindowProps) {
  const socket = useSocket();
  const [text, setText] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const worldId = view.world.id;

  useEffect(() => {
    socket
      .query<{ text: string }>({ kind: "notebook", worldId })
      .then((r) => setText(r.text))
      .catch(() => setText(""));
  }, [socket, worldId]);

  async function save() {
    if (text === null) return;
    setProblem(null);
    setSaved(false);
    const r = await socket.notebookSave(worldId, text);
    if (r.ok) setSaved(true);
    else setProblem(r.reason);
  }

  const count = text === null ? 0 : wordsIn(text);
  return (
    <WindowFrame entry={entry} title="Your notebook" subtitle="Private training text" wide>
      {text === null ? (
        <p className="text-sm text-muted">Opening your notebook…</p>
      ) : (
        <>
          <textarea
            className={`${INPUT} player-text min-h-64`}
            aria-label="Notebook text"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setSaved(false);
            }}
          />
          <p
            className={`mt-1 text-xs ${count > MAX_NOTEBOOK_WORDS ? "text-danger" : "text-muted"}`}
          >
            {`${formatCount(count)} / ${formatCount(MAX_NOTEBOOK_WORDS)} words`}
          </p>
          {problem && (
            <p role="alert" className="mt-2 text-sm text-danger">
              {problem}
            </p>
          )}
          {saved && (
            <p role="status" className="mt-2 text-sm text-good">
              Saved. Only your company's AI reads this.
            </p>
          )}
          <Button className="mt-4" onClick={() => void save()}>
            Save
          </Button>
        </>
      )}
    </WindowFrame>
  );
}
