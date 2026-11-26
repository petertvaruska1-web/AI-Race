import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Lobby } from "../src/screens/Lobby.tsx";
import { useSession } from "../src/store/session.ts";
import { useLook } from "../src/theme/glass.ts";
import { liveGame } from "./mockSocket.ts";
import { apply, command, LOGO, PALETTE, testWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
});

function preview() {
  return within(screen.getByTestId("preview"));
}

describe("founding a company", () => {
  it("shows the chosen palette and symbol in the preview", () => {
    render(<Lobby view={viewOf(testWorld())} />);
    fireEvent.click(screen.getByRole("button", { name: /Sunset/ }));
    fireEvent.click(screen.getByRole("button", { name: "rocket" }));
    const logo = preview().getByRole("img", { name: "Your logo" });
    expect(logo.getAttribute("data-symbol")).toBe("rocket");
    expect(logo.innerHTML).toContain("#ff7a45");
  });

  it("founds in a deep main color without a warning: the glass turns light", () => {
    render(<Lobby view={viewOf(testWorld())} />);
    fireEvent.change(screen.getByPlaceholderText("Acme Intelligence"), {
      target: { value: "Acme" },
    });
    fireEvent.change(screen.getByRole("slider", { name: "Main color brightness" }), {
      target: { value: "18" },
    });
    expect(screen.getByText(/Your lab glass is light/)).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(
      (screen.getByRole("button", { name: "Found company" }) as HTMLButtonElement).disabled,
    ).toBe(false);
  });

  it("sends exactly the chosen name, emblem and palette, from the keyboard too", async () => {
    render(<Lobby view={viewOf(testWorld())} />);
    fireEvent.change(screen.getByPlaceholderText("Acme Intelligence"), {
      target: { value: "  Acme Labs " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Letters" }));
    fireEvent.click(screen.getByRole("button", { name: /Forest/ }));
    screen.getByRole("button", { name: "Found company" }).focus();
    await userEvent.keyboard("{Enter}");
    await waitFor(() => expect(game.ws.sentOf("command")).toHaveLength(1));
    expect(game.ws.sentOf("command")[0]?.cmd).toEqual({
      type: "createCompany",
      name: "  Acme Labs ",
      logo: {
        shape: "hexagon",
        symbol: null,
        initials: "AL",
        emblem: {
          frame: "corners",
          corners: 6,
          turn: 0,
          roundness: 0,
          rings: 2,
          ringWeight: 3,
          glow: 40,
          symbol: null,
          letters: "AL",
          symbolSize: 100,
          pattern: "rays",
          finish: "metal",
        },
      },
      palette: { primary: "#4caf50", secondary: "#c5e384" },
    });
  });

  it("gives a name without letters the letter A", async () => {
    render(<Lobby view={viewOf(testWorld())} />);
    fireEvent.change(screen.getByPlaceholderText("Acme Intelligence"), {
      target: { value: "!!!" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Letters" }));
    fireEvent.click(screen.getByRole("button", { name: "Found company" }));
    await waitFor(() => expect(game.ws.sentOf("command")).toHaveLength(1));
    const cmd = game.ws.sentOf("command")[0]?.cmd as { logo: { emblem: { letters: string } } };
    expect(cmd.logo.emblem.letters).toBe("A");
  });

  it("keeps a hostile company name as plain text", () => {
    const { container } = render(<Lobby view={viewOf(testWorld())} />);
    const hostile = "<img src=x onerror=alert(1)>";
    const plate = screen.getByPlaceholderText("Acme Intelligence") as HTMLInputElement;
    fireEvent.change(plate, { target: { value: hostile } });
    expect(plate.value).toBe(hostile);
    expect(container.querySelector("img")).toBeNull();
  });

  it("waits for a name before the lever can be pulled", () => {
    render(<Lobby view={viewOf(testWorld())} />);
    expect(
      (screen.getByRole("button", { name: "Found company" }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(screen.getByText("Name your company first")).toBeTruthy();
  });

  it("shows the server's reason when founding is refused, and lets the lever up again", async () => {
    render(<Lobby view={viewOf(testWorld())} />);
    fireEvent.change(screen.getByPlaceholderText("Acme Intelligence"), {
      target: { value: "Acme" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Found company" }));
    await waitFor(() => expect(game.ws.sentOf("command")).toHaveLength(1));
    const cid = game.ws.sentOf("command")[0]?.cid;
    game.ws.receive({
      v: 1,
      type: "reject",
      cid,
      reason: "Another company already uses that name.",
    });
    expect(await screen.findByText("Another company already uses that name.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Found company" }));
    await waitFor(() => expect(game.ws.sentOf("command")).toHaveLength(2));
  });

  it("powers the lab glass up only once the server founds the company", async () => {
    render(<Lobby view={viewOf(testWorld())} />);
    fireEvent.change(screen.getByPlaceholderText("Acme Intelligence"), {
      target: { value: "Acme" },
    });
    const before = useLook.getState().power;
    fireEvent.click(screen.getByRole("button", { name: "Found company" }));
    await waitFor(() => expect(game.ws.sentOf("command")).toHaveLength(1));
    expect(useLook.getState().power).toBe(before);
    const cid = game.ws.sentOf("command")[0]?.cid;
    game.ws.receive({ v: 1, type: "ack", cid, duplicate: false });
    await waitFor(() => expect(useLook.getState().power).toBe(before + 1));
  });
});

describe("opening for business", () => {
  it("lets the host start the world once the company exists", async () => {
    const s = apply(
      testWorld(),
      command("u1", { type: "createCompany", name: "Acme", logo: LOGO, palette: PALETTE }, "c1"),
    );
    render(<Lobby view={viewOf(s)} />);
    expect(screen.getByText("Acme")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Open for business" }));
    await waitFor(() => expect(game.ws.sentOf("command")[0]?.cmd).toEqual({ type: "startWorld" }));
  });
});

describe("inviting friends", () => {
  it("lists the players with who is online, and copies the invite link", async () => {
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    let s = apply(
      testWorld(),
      command("u1", { type: "createCompany", name: "Acme", logo: LOGO, palette: PALETTE }, "c1"),
    );
    s = apply(s, {
      kind: "system",
      event: { type: "memberJoined", userId: "u2", displayName: "Ben" },
    });
    s = apply(s, { kind: "system", event: { type: "presence", userId: "u2", online: false } });
    render(<Lobby view={viewOf(s)} />);
    const players = within(screen.getByRole("list", { name: "Players" }));
    expect(players.getAllByRole("listitem").map((li) => li.textContent)).toEqual([
      expect.stringContaining("online"),
      expect.stringContaining("Ben"),
    ]);
    expect(players.getByText("Ben").closest("li")?.textContent).toContain("away");
    const code = s.inviteCode;
    expect(screen.getByText(code)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Copy link" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(`${location.origin}/join/${code}`));
    expect(await screen.findByText("Copied")).toBeTruthy();
  });
});
