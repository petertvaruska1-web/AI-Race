import type { ReactNode } from "react";

export type ChatMessage = { role: "user" | "ai"; text: string };

/**
 * A conversation as terminal lines (design §10, the Playground): who speaks, then their words.
 * Model text is shown exactly as written, as text with its line breaks (R1, spec §11);
 * `streaming` is the reply arriving now ("" while it thinks), with a caret after it.
 */
export function ChatLog({
  messages,
  streaming,
  aiName,
  coach,
}: {
  messages: ChatMessage[];
  streaming: string | null;
  aiName: string;
  /** Coaching under each stored AI reply that answers a message (your own AIs only). */
  coach?: (idx: number) => ReactNode;
}) {
  return (
    <ol className="chat-log" aria-live="polite" data-testid="chat-log">
      {messages.map((m, i) => (
        <Bubble
          // biome-ignore lint/suspicious/noArrayIndexKey: messages are only ever appended, so a position never changes hands
          key={`${i}:${m.role}`}
          who={m.role === "user" ? "You" : aiName}
          mine={m.role === "user"}
          below={
            coach && m.role === "ai" && messages[i - 1]?.role === "user" ? coach(i) : undefined
          }
        >
          {m.text}
        </Bubble>
      ))}
      {streaming !== null && (
        <Bubble who={aiName} mine={false}>
          {streaming === "" ? <span className="chat-line__thinking">Thinking…</span> : streaming}
          <span className="chat-caret" aria-hidden="true" />
        </Bubble>
      )}
    </ol>
  );
}

function Bubble({
  who,
  mine,
  children,
  below,
}: {
  who: string;
  mine: boolean;
  children: ReactNode;
  below?: ReactNode;
}) {
  return (
    <li className={`chat-line ${mine ? "chat-line--you" : "chat-line--ai"}`}>
      <span className="chat-line__who">{who}</span>
      <p className="player-text chat-line__text">{children}</p>
      {below}
    </li>
  );
}
