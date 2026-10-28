import type { WorldView } from "@airace/protocol";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { formatCount, formatMoney } from "../src/format.ts";
import { Company } from "../src/screens/Company.tsx";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import {
  apply,
  feedback,
  launchedWorld,
  runningWorld,
  SCORES,
  trainedWorld,
  viewOf,
} from "./views.ts";

beforeEach(async () => {
  useSession.setState({ socket: (await liveGame()).socket });
});

describe("company screen", () => {
  it("shows a launched AI's customers and the company's cash", () => {
    const view = viewOf(launchedWorld());
    const company = view.me.company;
    expect(company?.customers.total).toBeGreaterThan(0);
    render(<Company view={view} />);
    expect(screen.getByText(formatCount(company?.customers.total ?? -1))).toBeTruthy();
    expect(screen.getByText(formatMoney(company?.cash ?? -1))).toBeTruthy();
  });

  it("quotes a customer's post exactly as the AI wrote it, as text", () => {
    const s = apply(
      launchedWorld(),
      feedback("w9", [
        ["students", 0.9, "<b>x</b> <|end|>"],
        ["developers", 0.1, "No idea."],
      ]),
    );
    const { container } = render(<Company view={viewOf(s)} />);
    expect(screen.getByText("<b>x</b> <|end|>")).toBeTruthy();
    expect(container.querySelector("blockquote b")).toBeNull();
  });

  it("folds a long reply behind Show more, keeping all of it", () => {
    const long = "x".repeat(2000);
    const s = apply(
      launchedWorld(),
      feedback("w9", [
        ["students", 0.9, long],
        ["developers", 0.1, "No idea."],
      ]),
    );
    const { container } = render(<Company view={viewOf(s)} />);
    expect(screen.getByRole("button", { name: "Show more" })).toBeTruthy();
    expect(container.textContent).toContain(long);
  });

  it("shows the spark and asks for a first AI before there is one", () => {
    render(<Company view={viewOf(runningWorld())} />);
    expect(screen.getByRole("img", { name: /spark/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Train your first AI" })).toBeTruthy();
  });

  it("holds the launch until the newest AI has been scored", () => {
    render(<Company view={viewOf(trainedWorld())} />);
    const launch = screen.getByRole("button", { name: "Launch" }) as HTMLButtonElement;
    expect(launch.disabled).toBe(true);
    expect(launch.title).toMatch(/Wait until your AI has been scored/);
    expect(screen.getByText(/Wait until your AI has been scored/)).toBeTruthy();
  });
});

describe("company screen clicks", () => {
  beforeEach(() => useWindows.getState().closeAll());
  const withWindows = (view: WorldView) =>
    render(
      <>
        <Company view={view} />
        <WindowHost view={view} />
      </>,
    );

  it("opens the AI's window from the orb", async () => {
    const view = viewOf(trainedWorld(SCORES));
    const label = view.me.company?.versions[0]?.label;
    withWindows(view);
    await userEvent.click(screen.getByRole("button", { name: `About your AI ${label}` }));
    expect(await screen.findByRole("dialog", { name: `Your AI ${label}` })).toBeTruthy();
  });

  it("opens customers, a group, money and a product from where they show", async () => {
    withWindows(viewOf(launchedWorld()));
    for (const [button, dialog] of [
      ["Open customers", "Customers"],
      ["Open Students", "Students"],
      ["Open money", "Money"],
      ["Open Spark", "Spark"],
    ] as const) {
      await userEvent.click(screen.getByRole("button", { name: button }));
      expect(await screen.findByRole("dialog", { name: dialog })).toBeTruthy();
      await userEvent.click(screen.getByRole("button", { name: "Close all windows" }));
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    }
  });

  it("opens a post from the feed", async () => {
    const s = apply(
      launchedWorld(),
      feedback("w9", [
        ["students", 0.9, "Paris, of course."],
        ["developers", 0.1, "No idea."],
      ]),
    );
    withWindows(viewOf(s));
    const cards = screen.getAllByRole("button", { name: /^Open the post by/ });
    await userEvent.click(cards[0] as HTMLElement);
    const w = await screen.findByRole("dialog");
    expect(within(w).getByRole("button", { name: /^Open (Students|Developers)$/ })).toBeTruthy();
  });
});
