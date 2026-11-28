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
import { launchedWorld, runningWorld, SCORES, ticks, trainedWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
});

const commands = () => game.ws.sentOf("command");

describe("the Money bench's warning", () => {
  it("can be closed so it never hides the controls under it", async () => {
    render(<Money view={viewOf(launchedWorld())} />);
    fireEvent.click(screen.getByRole("button", { name: "Retire Spark" }));
    await waitFor(() => expect(commands()).toHaveLength(1));
    const cid = commands()[0]?.cid;
    game.ws.receive({ v: 1, type: "reject", cid, reason: "Move its plans to another AI first." });
    expect((await screen.findByRole("alert")).textContent).toContain("Move its plans");
    fireEvent.click(screen.getByRole("button", { name: "Close the warning" }));
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

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
      keynote: null,
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

  it("sets a price with its knob, within its bounds", async () => {
    render(<Money view={viewOf(launchedWorld())} />);
    const knob = screen.getByRole("slider", { name: "Personal price" });
    const before = Number(knob.getAttribute("aria-valuenow"));
    fireEvent.keyDown(knob, { key: "ArrowUp" });
    fireEvent.click(screen.getByRole("button", { name: "Set the personal price" }));
    await waitFor(() => expect(commands()).toHaveLength(1));
    expect(commands()[0]?.cmd).toEqual({ type: "setPrice", plan: "personal", price: before + 1 });
    fireEvent.keyDown(knob, { key: "End" });
    expect(knob.getAttribute("aria-valuenow")).toBe("200");
    fireEvent.keyDown(knob, { key: "ArrowUp" });
    expect(knob.getAttribute("aria-valuenow")).toBe("200");
  });

  it("picks each plan's AI on keys, Closed among them", async () => {
    render(<Money view={viewOf(launchedWorld())} />);
    const keys = screen.getByRole("group", { name: "AI for the business plan" });
    await userEvent.click(within(keys).getByRole("radio", { name: "Spark" }));
    await waitFor(() => expect(commands()).toHaveLength(1));
    expect(commands()[0]?.cmd).toMatchObject({ type: "setPlan", plan: "business" });
    expect(within(keys).getByRole("radio", { name: "Closed" })).toBeTruthy();
  });

  it("shows a new company's Money bench with every module and its empty states", () => {
    render(<Money view={viewOf(runningWorld())} />);
    for (const name of ["Coming in", "Going out", "Report cards", "Computers", "Investment fund"])
      expect(screen.getByRole("region", { name })).toBeTruthy();
    for (const name of ["Profit by month", "Your AIs on sale", "Plans and prices"])
      expect(screen.getByRole("heading", { name })).toBeTruthy();
    expect(screen.getByText("Nothing on sale yet. Launch an AI to get customers.")).toBeTruthy();
    expect(
      screen.getByText("Your first report card comes at the end of the month after you launch."),
    ).toBeTruthy();
  });

  it("racks a year of report cards inside their own display", () => {
    const view = viewOf(ticks(launchedWorld(), 70));
    const company = view.me.company;
    const first = company?.reports[0];
    if (!company || !first) throw new Error("no report card");
    company.reports = Array.from({ length: 12 }, (_, i) => ({
      ...first,
      month: `2028-${String(i + 1).padStart(2, "0")}`,
    }));
    render(<Money view={view} />);
    const rack = screen.getByRole("region", { name: "Report cards, newest first" });
    expect(rack.className).toContain("display--scroll");
    expect(within(rack).getAllByRole("button")).toHaveLength(12);
    expect(within(rack).getByRole("button", { name: `${monthName("2028-12")}: ${first.grade}` }));
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
      ["Cash day by day", "Money"],
    ] as const) {
      await userEvent.click(screen.getByRole("button", { name: "Close all windows" }));
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
      await userEvent.click(screen.getByRole("button", { name: button }));
      expect(await screen.findByRole("dialog", { name: dialog })).toBeTruthy();
    }
  });
});
