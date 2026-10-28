import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { AppShell } from "../src/components/AppShell.tsx";
import { monthName } from "../src/format.ts";
import { Company } from "../src/screens/Company.tsx";
import { Money } from "../src/screens/Money.tsx";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { apply, feedback, launchedWorld, ticks, viewOf } from "./views.ts";

beforeEach(async () => {
  useSession.setState({ socket: (await liveGame()).socket });
  useWindows.getState().closeAll();
});

const HOSTILE = "<b>bold</b> <|end|>";

/** A world whose first report card (February) quotes a hostile reply as its worst moment. */
function graded() {
  const s = apply(
    launchedWorld(),
    feedback("w9", [
      ["students", 0.9, "Paris, of course."],
      ["developers", 0.1, HOSTILE],
    ]),
  );
  return ticks(s, 30);
}

describe("report cards on screen", () => {
  it("open from their tile, with the grade and the month's posts quoted exactly", async () => {
    const view = viewOf(graded());
    const card = view.me.company?.reports.at(-1);
    if (!card) throw new Error("no report card");
    const { container } = render(
      <>
        <Money view={view} />
        <WindowHost view={view} />
      </>,
    );
    const panel = screen.getByRole("region", { name: "Report cards" });
    await userEvent.click(
      within(panel).getByRole("button", { name: `${monthName(card.month)}: ${card.grade}` }),
    );
    const w = await screen.findByRole("dialog", { name: `${monthName(card.month)} report card` });
    expect(within(w).getByTestId("grade").textContent).toBe(card.grade);
    expect(within(w).getByText("Paris, of course.")).toBeTruthy();
    expect(within(w).getByText(HOSTILE)).toBeTruthy();
    expect(container.ownerDocument.querySelector("[role=dialog] blockquote b")).toBeNull();
  });

  it("pop up as a toast when a new one arrives, and only then", async () => {
    const before = viewOf(ticks(launchedWorld(), 1));
    const after = viewOf(graded());
    const card = after.me.company?.reports.at(-1);
    if (!card) throw new Error("no report card");
    const shell = (v: typeof before) => (
      <AppShell view={v} tab="company">
        <p>Screen</p>
      </AppShell>
    );
    const { rerender } = render(shell(before));
    expect(screen.queryByText(/report card:/)).toBeNull();
    rerender(shell(after));
    const toast = await screen.findByText(
      `Your ${monthName(card.month)} report card: ${card.grade}`,
    );
    await userEvent.click(toast);
    expect(
      await screen.findByRole("dialog", { name: `${monthName(card.month)} report card` }),
    ).toBeTruthy();
    // Seen once, it does not come back.
    rerender(shell(after));
    await waitFor(() => expect(screen.queryByText(/report card:/)).toBeNull());
  });

  it("show the latest grade on the Company screen, which opens the card", async () => {
    const view = viewOf(graded());
    const card = view.me.company?.reports.at(-1);
    if (!card) throw new Error("no report card");
    render(
      <>
        <Company view={view} />
        <WindowHost view={view} />
      </>,
    );
    await userEvent.click(screen.getByRole("button", { name: `Last report card: ${card.grade}` }));
    expect(
      await screen.findByRole("dialog", { name: `${monthName(card.month)} report card` }),
    ).toBeTruthy();
  });
});
