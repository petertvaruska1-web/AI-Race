import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { formatMoney } from "../src/format.ts";
import { LaunchDialog } from "../src/screens/LaunchDialog.tsx";
import { Money } from "../src/screens/Money.tsx";
import { useSession } from "../src/store/session.ts";
import { liveGame } from "./mockSocket.ts";
import { launchedWorld, SCORES, trainedWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
});

const commands = () => game.ws.sentOf("command");

describe("launch dialog", () => {
  it("sends the chosen AI, name, plans and countdown", async () => {
    render(<LaunchDialog view={viewOf(trainedWorld(SCORES))} onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText("Product name"), { target: { value: "Spark" } });
    fireEvent.click(screen.getByRole("button", { name: "Launch" }));
    await waitFor(() => expect(commands()).toHaveLength(1));
    expect(commands()[0]?.cmd).toEqual({
      type: "launchProduct",
      versionId: "v1",
      name: "Spark",
      plans: ["free", "personal"],
      countdownDays: 5,
    });
  });

  it("holds a name longer than 24 characters", () => {
    render(<LaunchDialog view={viewOf(trainedWorld(SCORES))} onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText("Product name"), { target: { value: "x".repeat(25) } });
    const launch = screen.getByRole("button", { name: "Launch" }) as HTMLButtonElement;
    expect(launch.disabled).toBe(true);
    expect(screen.getByText(/2–24 characters/)).toBeTruthy();
  });
});

describe("money screen", () => {
  it("shows this month's income from the personal plan", () => {
    const view = viewOf(launchedWorld());
    const personal = view.me.company?.ledger.current.income.personal ?? 0;
    expect(personal).toBeGreaterThan(0);
    render(<Money view={view} />);
    expect(screen.getAllByText(formatMoney(personal)).length).toBeGreaterThan(0);
  });

  it("sets a price within its bounds and refuses one outside them", async () => {
    render(<Money view={viewOf(launchedWorld())} />);
    const price = screen.getByLabelText("Personal price");
    const set = screen.getByRole("button", { name: "Set the personal price" });
    fireEvent.change(price, { target: { value: "500" } });
    fireEvent.click(set);
    expect(screen.getByRole("alert").textContent).toMatch(/\$1–\$200/);
    expect(commands()).toHaveLength(0);
    fireEvent.change(price, { target: { value: "15" } });
    fireEvent.click(set);
    await waitFor(() => expect(commands()).toHaveLength(1));
    expect(commands()[0]?.cmd).toEqual({ type: "setPrice", plan: "personal", price: 15 });
  });

  it("shows the server's reason when an AI that powers a plan is retired", async () => {
    render(<Money view={viewOf(launchedWorld())} />);
    fireEvent.click(screen.getByRole("button", { name: "Retire Spark" }));
    await waitFor(() => expect(commands()).toHaveLength(1));
    game.ws.receive({
      v: 1,
      type: "reject",
      cid: commands()[0]?.cid,
      reason: "Move its plans to another AI first.",
    });
    expect(await screen.findByText("Move its plans to another AI first.")).toBeTruthy();
  });
});
