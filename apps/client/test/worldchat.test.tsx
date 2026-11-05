import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { WorldChat } from "../src/components/WorldChat.tsx";
import { useSession } from "../src/store/session.ts";
import { liveGame } from "./mockSocket.ts";
import { apply, command, LOGO, runningWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
});

function world() {
  let s = runningWorld();
  s = apply(s, {
    kind: "system",
    event: { type: "memberJoined", userId: "u2", displayName: "Ben" },
  });
  return apply(
    s,
    command(
      "u2",
      {
        type: "createCompany",
        name: "Bolt",
        logo: LOGO,
        palette: { primary: "#e63946", secondary: "#ffb020" },
      },
      "c2",
    ),
  );
}

function push(worldId: string, line: { name: string; companyId: string | null; text: string }) {
  game.ws.receive({ v: 1, type: "stream", sub: `worldchat:${worldId}`, data: { ...line, at: 1 } });
}

describe("the world chat panel", () => {
  it("sends a message, shows the reply in its company's color, and shows a refusal", async () => {
    const view = viewOf(world());
    render(<WorldChat view={view} />);
    fireEvent.click(screen.getByRole("button", { name: /World chat/ }));
    fireEvent.change(screen.getByLabelText("Message"), { target: { value: "Good luck!" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() =>
      expect(game.ws.sentOf("worldChat").at(-1)).toMatchObject({
        worldId: view.world.id,
        text: "Good luck!",
      }),
    );
    const first = game.ws.sentOf("worldChat").at(-1)?.cid;
    game.ws.receive({ v: 1, type: "ack", cid: first, duplicate: false });
    await waitFor(() =>
      expect((screen.getByLabelText("Message") as HTMLInputElement).value).toBe(""),
    );
    push(view.world.id, { name: "Ben", companyId: "c2", text: "You too" });
    const name = await screen.findByText("Ben");
    expect(name.getAttribute("style")).toContain("#e63946");
    expect(screen.getByText("You too")).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Message"), { target: { value: "www.example.com" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(game.ws.sentOf("worldChat")).toHaveLength(2));
    const cid = game.ws.sentOf("worldChat").at(-1)?.cid;
    game.ws.receive({
      v: 1,
      type: "reject",
      cid,
      reason: "Messages can't hold links, email addresses or phone numbers.",
    });
    expect(
      await screen.findByText("Messages can't hold links, email addresses or phone numbers."),
    ).toBeTruthy();
  });

  it("counts unread messages while it is closed", async () => {
    const view = viewOf(world());
    render(<WorldChat view={view} />);
    push(view.world.id, { name: "Ben", companyId: "c2", text: "one" });
    push(view.world.id, { name: "Ben", companyId: "c2", text: "two" });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /World chat/ }).textContent).toContain("2"),
    );
    fireEvent.click(screen.getByRole("button", { name: /World chat/ }));
    expect(screen.getByRole("button", { name: /World chat/ }).textContent).not.toContain("2");
  });

  it("shows markup as plain text", async () => {
    const view = viewOf(world());
    const { container } = render(<WorldChat view={view} />);
    fireEvent.click(screen.getByRole("button", { name: /World chat/ }));
    push(view.world.id, { name: "Ben", companyId: "c2", text: "<b>bold</b><img src=x>" });
    expect(await screen.findByText("<b>bold</b><img src=x>")).toBeTruthy();
    expect(container.querySelector("b, img")).toBeNull();
  });

  it("loads the earlier messages", async () => {
    const view = viewOf(world());
    render(<WorldChat view={view} />);
    const q = await waitFor(() => {
      const sent = game.ws.sentOf("query").find((m) => m.q.kind === "worldChat");
      if (!sent) throw new Error("not yet");
      return sent;
    });
    game.ws.receive({
      v: 1,
      type: "queryResult",
      id: q.id,
      ok: true,
      data: { messages: [{ name: "Ana", companyId: "c1", text: "Earlier", at: 0 }] },
    });
    fireEvent.click(screen.getByRole("button", { name: /World chat/ }));
    expect(await screen.findByText("Earlier")).toBeTruthy();
  });
});
