import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { Playground, TOO_LONG } from "../src/screens/Playground.tsx";
import { usePlayground } from "../src/store/playground.ts";
import { useSession } from "../src/store/session.ts";
import { liveGame } from "./mockSocket.ts";
import { trainedWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;
const THREAD = "11111111-1111-4111-8111-111111111111";

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  usePlayground.setState({ versionId: null });
});

/** Types a message, sends it and acks it as thread THREAD; resolves once the reply stream is followed. */
async function sendAndAck(text: string) {
  fireEvent.change(screen.getByLabelText("Your message"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
  await waitFor(() => expect(game.ws.sentOf("chat")).toHaveLength(1));
  const chat = game.ws.sentOf("chat")[0];
  expect(chat).toMatchObject({ worldId: "w1", threadId: null, versionId: "v1", text });
  game.ws.receive({
    v: 1,
    type: "ack",
    cid: chat?.cid,
    duplicate: false,
    data: { threadId: THREAD },
  });
  await waitFor(() =>
    expect(game.ws.sentOf("subscribe").map((s) => s.sub)).toEqual([`chat:w1:${THREAD}`]),
  );
}

const stream = (data: unknown) =>
  game.ws.receive({ v: 1, type: "stream", sub: `chat:w1:${THREAD}`, data });

describe("playground", () => {
  it("sends a message and shows the reply as it streams in", async () => {
    render(<Playground view={viewOf(trainedWorld())} />);
    await sendAndAck("Hello there");
    expect(screen.getByText("Hello there")).toBeTruthy();
    expect(screen.getByText("Thinking…")).toBeTruthy();
    stream({ kind: "chunk", text: "Hi" });
    stream({ kind: "chunk", text: " there" });
    expect(await screen.findByText("Hi there")).toBeTruthy();
    stream({ kind: "done", text: "Hi there!" });
    expect(await screen.findByText("Hi there!")).toBeTruthy();
    expect(screen.queryByText("Thinking…")).toBeNull();
    expect((screen.getByRole("button", { name: "Send" }) as HTMLButtonElement).disabled).toBe(true); // empty draft
  });

  it("asks for a shorter message over 500 characters", () => {
    render(<Playground view={viewOf(trainedWorld())} />);
    fireEvent.change(screen.getByLabelText("Your message"), { target: { value: "x".repeat(501) } });
    expect(screen.getByText(TOO_LONG)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Send" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows the AI's markup as plain text", async () => {
    const { container } = render(<Playground view={viewOf(trainedWorld())} />);
    await sendAndAck("Say something bold");
    stream({ kind: "done", text: "<b>x</b>" });
    expect(await screen.findByText("<b>x</b>")).toBeTruthy();
    expect(container.querySelector("b")).toBeNull();
  });

  it("offers to try again when the reply fails", async () => {
    render(<Playground view={viewOf(trainedWorld())} />);
    await sendAndAck("Hello");
    stream({ kind: "failed", reason: "The AI couldn't answer right now. Try again." });
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }));
    await waitFor(() => expect(game.ws.sentOf("chat")).toHaveLength(2));
    expect(game.ws.sentOf("chat")[1]).toMatchObject({ threadId: THREAD, text: "Hello" });
  });

  it("warns that an AI that never read a conversation may just continue the text", () => {
    const { unmount } = render(<Playground view={viewOf(trainedWorld(undefined, 0))} />);
    expect(screen.getByText(/never saw a conversation/)).toBeTruthy();
    unmount();
    render(<Playground view={viewOf(trainedWorld())} />);
    expect(screen.queryByText(/never saw a conversation/)).toBeNull();
  });
});
