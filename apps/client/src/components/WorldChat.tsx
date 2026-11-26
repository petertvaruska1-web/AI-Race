import type { WorldView } from "@airace/protocol";
import { type FormEvent, useEffect, useRef, useState } from "react";
import { Display } from "../bench/Display.tsx";
import { PushButton } from "../bench/PushButton.tsx";
import { useSocket } from "../store/session.ts";
import { readableOn } from "../theme/glass.ts";
import { ABOVE_BOTTOM_BAR } from "./bottomBar.ts";

type Line = { name: string; companyId: string | null; text: string; at: number };
const KEPT = 200;

/**
 * The world chat (Stage 2): an intercom at the foot of the station column, closed at first,
 * with an unread count. Each name shows in its company's color; messages are shown as plain text.
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

  // Names sit in a dark display, so each company's color is made readable on dark (A3).
  const colorOf = (companyId: string | null) => {
    const color = view.companies.find((c) => c.id === companyId)?.palette.primary;
    return color ? readableOn(color, "dark") : "var(--muted)";
  };

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
      className="fixed left-[26px] z-30 max-w-[calc(100vw-2rem)]"
      style={{ bottom: ABOVE_BOTTOM_BAR }}
    >
      {open && (
        <div className="module module--heavy mb-2 flex max-h-96 w-80 max-w-full flex-col">
          <Display scroll label="World chat" className="mb-2 min-h-24 flex-1">
            <ul className="space-y-1 text-sm">
              {lines.length === 0 && <li className="text-muted">No messages yet. Say hello!</li>}
              {lines.map((l, i) => (
                <li
                  key={`${l.at}-${l.name}-${i.toString(36)}`}
                  ref={i === lines.length - 1 ? endRef : null}
                >
                  <span
                    className="player-text font-semibold"
                    style={{ color: colorOf(l.companyId) }}
                  >
                    {l.name}
                  </span>{" "}
                  <span className="player-text whitespace-pre-wrap break-words">{l.text}</span>
                </li>
              ))}
            </ul>
          </Display>
          <form onSubmit={send} className="flex gap-2">
            <input
              aria-label="Message"
              className="display-input min-w-0 flex-1 py-1.5 text-sm"
              value={draft}
              maxLength={300}
              onChange={(e) => setDraft(e.target.value)}
            />
            <PushButton type="submit" disabled={busy || draft.trim().length === 0}>
              Send
            </PushButton>
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
        className="intercom-key"
      >
        <span className="intercom-key__grille" aria-hidden="true" />
        World chat
        {unread > 0 && !open && (
          <span className="display intercom-key__count dotface dotface--sm">{unread}</span>
        )}
      </button>
    </div>
  );
}
