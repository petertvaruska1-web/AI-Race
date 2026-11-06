import type { WorldView } from "@airace/protocol";
import { type FormEvent, useEffect, useRef, useState } from "react";
import { useSocket } from "../store/session.ts";
import { ABOVE_BOTTOM_BAR } from "./bottomBar.ts";

type Line = { name: string; companyId: string | null; text: string; at: number };
const KEPT = 200;

/**
 * The world chat (Stage 2): a panel in the bottom-left corner, closed at first, with an unread
 * count. Each name shows in its company's color; messages are shown as plain text.
 */
export function WorldChat({ view }: { view: WorldView }) {
  const socket = useSocket();
  const worldId = view.world.id;
  const [open, setOpen] = useState(false);
  const [lines, setLines] = useState<Line[]>([]);
  const [unread, setUnread] = useState(0);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const openRef = useRef(open);
  openRef.current = open;
  const myCompany = view.me.company?.id ?? null;
  const myName = view.members.find((m) => m.userId === view.me.userId)?.displayName ?? null;
  const endRef = useRef<HTMLLIElement | null>(null);

  useEffect(() => {
    let alive = true;
    socket
      .query<{ messages: Line[] }>({ kind: "worldChat", worldId })
      .then((r) => alive && setLines((now) => [...r.messages, ...now].slice(-KEPT)))
      .catch(() => {});
    const stop = socket.subscribe(`worldchat:${worldId}`, (data) => {
      const line = data as Line;
      setLines((now) => [...now, line].slice(-KEPT));
      const mine = myCompany !== null ? line.companyId === myCompany : line.name === myName;
      if (!openRef.current && !mine) setUnread((n) => n + 1);
    });
    return () => {
      alive = false;
      stop();
    };
  }, [socket, worldId, myCompany, myName]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll on each new line
  useEffect(() => {
    if (open) endRef.current?.scrollIntoView?.({ block: "end" });
  }, [open, lines.length]);

  const colorOf = (companyId: string | null) =>
    view.companies.find((c) => c.id === companyId)?.palette.primary ?? "var(--muted)";

  async function send(e: FormEvent) {
    e.preventDefault();
    const text = draft.trim();
    if (!text) return;
    setBusy(true);
    setError(null);
    const r = await socket.worldChat(worldId, text);
    setBusy(false);
    if (r.ok) setDraft("");
    else setError(r.reason);
  }

  return (
    <div
      className="fixed left-4 z-30 w-80 max-w-[calc(100vw-2rem)]"
      style={{ bottom: ABOVE_BOTTOM_BAR }}
    >
      {open && (
        <div className="window mb-2 flex max-h-96 flex-col p-3">
          <ul className="mb-2 flex-1 space-y-1 overflow-y-auto text-sm">
            {lines.length === 0 && <li className="text-muted">No messages yet. Say hello!</li>}
            {lines.map((l, i) => (
              <li
                key={`${l.at}-${l.name}-${i.toString(36)}`}
                ref={i === lines.length - 1 ? endRef : null}
              >
                <span className="player-text font-semibold" style={{ color: colorOf(l.companyId) }}>
                  {l.name}
                </span>{" "}
                <span className="player-text whitespace-pre-wrap break-words">{l.text}</span>
              </li>
            ))}
          </ul>
          <form onSubmit={send} className="flex gap-2">
            <input
              aria-label="Message"
              className="min-w-0 flex-1 rounded-lg border border-border bg-surface px-2 py-1 text-sm"
              value={draft}
              maxLength={300}
              onChange={(e) => setDraft(e.target.value)}
            />
            <button
              type="submit"
              disabled={busy || draft.trim().length === 0}
              className="rounded-lg bg-brand px-3 py-1 text-sm text-on-brand disabled:opacity-50"
            >
              Send
            </button>
          </form>
          {error && (
            <p role="alert" className="mt-1 text-xs text-danger">
              {error}
            </p>
          )}
        </div>
      )}
      <button
        type="button"
        onClick={() => {
          setOpen((o) => !o);
          setUnread(0);
        }}
        className="rounded-full border border-border bg-surface px-4 py-2 text-sm font-semibold shadow"
      >
        World chat
        {unread > 0 && !open && (
          <span className="ml-2 rounded-full bg-brand px-2 text-xs text-on-brand">{unread}</span>
        )}
      </button>
    </div>
  );
}
