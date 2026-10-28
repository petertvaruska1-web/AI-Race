import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { formatMoney, monthName } from "../src/format.ts";
import { profitOf } from "../src/money.ts";
import { LaunchDialog } from "../src/screens/LaunchDialog.tsx";
import { Money } from "../src/screens/Money.tsx";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { launchedWorld, SCORES, ticks, trainedWorld, viewOf } from "./views.ts";

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

  it("colors this month's costs against last month's as good or bad news", () => {
    const view = viewOf(launchedWorld());
    const ledger = view.me.company?.ledger;
    const sum = (r: Record<string, number> | undefined) =>
      Object.values(r ?? {}).reduce((a, b) => a + b, 0);
    const up = sum(ledger?.current.costs) > sum(ledger?.months.at(-1)?.costs);
    render(<Money view={view} />);
    const out = screen.getByRole("region", { name: "Going out" });
    const change = within(out).getByText(/against last month/).parentElement;
    expect(change?.className).toContain(up ? "text-danger" : "text-good");
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

describe("money screen clicks", () => {
  beforeEach(() => useWindows.getState().closeAll());

  it("opens a month from its bar, a category from its row and the money window", async () => {
    const view = viewOf(ticks(launchedWorld(), 40));
    render(
      <>
        <Money view={view} />
        <WindowHost view={view} />
      </>,
    );
    const closed = view.me.company?.ledger.months.at(-1);
    if (!closed) throw new Error("no closed month");
    const months = screen.getByRole("group", { name: "Profit by month" });
    await userEvent.click(
      within(months).getByRole("button", {
        name: `${monthName(closed.month)}: ${formatMoney(profitOf(closed))}`,
      }),
    );
    expect(await screen.findByRole("dialog", { name: monthName(closed.month) })).toBeTruthy();
    for (const [button, dialog] of [
      ["Open Serving customers", "Serving customers"],
      ["Open Personal plans", "Personal plans"],
      ["Open money", "Money"],
    ] as const) {
      await userEvent.click(screen.getByRole("button", { name: "Close all windows" }));
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
      await userEvent.click(screen.getByRole("button", { name: button }));
      expect(await screen.findByRole("dialog", { name: dialog })).toBeTruthy();
    }
  });
});
