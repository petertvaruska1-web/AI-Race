import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { HARD_TO_READ } from "../src/components/PalettePicker.tsx";
import { Lobby } from "../src/screens/Lobby.tsx";
import { useSession } from "../src/store/session.ts";
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

  it("warns about a hard-to-read custom color without blocking", () => {
    render(<Lobby view={viewOf(testWorld())} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Acme" } });
    fireEvent.click(screen.getByRole("button", { name: "Custom colors" }));
    fireEvent.change(screen.getByLabelText("Main color"), { target: { value: "#202020" } });
    expect(screen.getByRole("alert").textContent).toBe(HARD_TO_READ);
    expect(
      (screen.getByRole("button", { name: "Found company" }) as HTMLButtonElement).disabled,
    ).toBe(false);
  });

  it("sends exactly the chosen name, logo and palette", async () => {
    render(<Lobby view={viewOf(testWorld())} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "  Acme Labs " } });
    fireEvent.click(screen.getByRole("button", { name: "Hexagon" }));
    fireEvent.click(screen.getByLabelText("Your initials"));
    fireEvent.change(screen.getByLabelText("Initials"), { target: { value: "a-l!x" } });
    fireEvent.click(screen.getByRole("button", { name: /Forest/ }));
    fireEvent.click(screen.getByRole("button", { name: "Found company" }));
    await waitFor(() => expect(game.ws.sentOf("command")).toHaveLength(1));
    expect(game.ws.sentOf("command")[0]?.cmd).toEqual({
      type: "createCompany",
      name: "  Acme Labs ",
      logo: { shape: "hexagon", symbol: null, initials: "AL" },
      palette: { primary: "#4caf50", secondary: "#c5e384" },
    });
  });

  it("shows a hostile company name as plain text", () => {
    const { container } = render(<Lobby view={viewOf(testWorld())} />);
    const hostile = "<img src=x onerror=alert(1)>";
    fireEvent.change(screen.getByRole("textbox"), { target: { value: hostile } });
    expect(preview().getByText(hostile)).toBeTruthy();
    expect(container.querySelector("img")).toBeNull();
  });

  it("shows the server's reason when founding is refused", async () => {
    render(<Lobby view={viewOf(testWorld())} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Acme" } });
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
