import type { Command, WorldView } from "@airace/protocol";
import { BALANCE_TEST } from "@airace/sim";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { AppShell } from "../src/components/AppShell.tsx";
import { formatMoney } from "../src/format.ts";
import { Company } from "../src/screens/Company.tsx";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { liveGame } from "./mockSocket.ts";
import { launchedWorld, ticks, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useWindows.getState().closeAll();
});

const commands = () => game.ws.sentOf("command").map((m) => m.cmd as Command);

/** A world where c1's cash ran out at the last month end, so a rescue is on offer. */
function broke(): WorldView {
  const s = structuredClone(launchedWorld());
  const c = s.companies.c1;
  if (!c) throw new Error("no company");
  c.cash = -5_000_000;
  const later = ticks(s, 30);
  const view = viewOf(later);
  if (!view.me.company?.rescue) throw new Error("no rescue offer");
  return view;
}

const shell = (view: WorldView) => (
  <AppShell view={view} tab="company">
    <p>Screen</p>
  </AppShell>
);

describe("rescue in the shell", () => {
  it("shows a banner while an offer waits, opening the two choices", async () => {
    const view = broke();
    const left = (view.me.company?.rescue?.deadlineDay ?? 0) - view.world.day;
    render(shell(view));
    expect(
      screen.getByText(`Your cash ran out. Choose a rescue within ${left} game days.`),
    ).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Choose a rescue" }));
    const w = await screen.findByRole("dialog", { name: "Choose a rescue" });
    const amount = BALANCE_TEST.economy.rescue.cushion - (view.me.company?.cash ?? 0);
    expect(within(w).getAllByText(formatMoney(amount)).length).toBe(2);
    expect(within(w).getByText("You keep 75%")).toBeTruthy();
    await userEvent.click(within(w).getByRole("button", { name: "Take the emergency loan" }));
    await waitFor(() => expect(commands()).toEqual([{ type: "chooseRescue", kind: "loan" }]));
  });

  it("can bring in an investor instead", async () => {
    render(shell(broke()));
    await userEvent.click(screen.getByRole("button", { name: "Choose a rescue" }));
    const w = await screen.findByRole("dialog", { name: "Choose a rescue" });
    await userEvent.click(within(w).getByRole("button", { name: "Bring in an investor" }));
    await waitFor(() => expect(commands()).toEqual([{ type: "chooseRescue", kind: "investor" }]));
  });

  it("has no banner without an offer", () => {
    render(shell(viewOf(launchedWorld())));
    expect(screen.queryByRole("button", { name: "Choose a rescue" })).toBeNull();
  });
});

describe("overload warning", () => {
  it("shows in the top bar while questions go unanswered, opening the computers", async () => {
    const view = viewOf(launchedWorld());
    const company = view.me.company;
    if (!company) throw new Error("no company");
    const busy: WorldView = {
      ...view,
      me: { ...view.me, company: { ...company, compute: { ...company.compute, served: 0.82 } } },
    };
    const { rerender } = render(shell(busy));
    await userEvent.click(
      screen.getByRole("button", { name: "Servers overloaded · 82% answered" }),
    );
    expect(await screen.findByRole("dialog", { name: "Computers" })).toBeTruthy();
    const calm: WorldView = {
      ...view,
      me: { ...view.me, company: { ...company, compute: { ...company.compute, served: 1 } } },
    };
    rerender(shell(calm));
    expect(screen.queryByRole("button", { name: /Servers overloaded/ })).toBeNull();
  });
});

describe("ownership", () => {
  it("shows the founders' share on the Company screen only once investors hold some", () => {
    const view = viewOf(launchedWorld());
    const { rerender } = render(<Company view={view} />);
    expect(screen.queryByText(/^You own/)).toBeNull();
    const shared: WorldView = {
      ...view,
      companies: view.companies.map((c) =>
        c.id === view.me.company?.id ? { ...c, ownership: 0.75 } : c,
      ),
    };
    rerender(<Company view={shared} />);
    expect(screen.getByText("You own 75%")).toBeTruthy();
  });
});
