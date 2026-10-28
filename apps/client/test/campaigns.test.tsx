import type { Command } from "@airace/protocol";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { formatMoney } from "../src/format.ts";
import { Market } from "../src/screens/Market.tsx";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { apply, command, launchedWorld, ticks, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useWindows.getState().closeAll();
});

const commands = () => game.ws.sentOf("command").map((m) => m.cmd as Command);

function market(view = viewOf(launchedWorld())) {
  render(
    <>
      <Market view={view} />
      <WindowHost view={view} />
    </>,
  );
  return view;
}

describe("ad campaigns on screen", () => {
  it("start from a window that sends exactly what the player chose", async () => {
    market();
    await userEvent.click(screen.getByRole("button", { name: "New ad campaign" }));
    const w = await screen.findByRole("dialog", { name: "New ad campaign" });
    await userEvent.selectOptions(within(w).getByLabelText("Where"), "pacific");
    await userEvent.selectOptions(within(w).getByLabelText("Who"), "developers");
    await userEvent.click(within(w).getByRole("button", { name: formatMoney(2000) }));
    fireEvent.change(within(w).getByLabelText("Days"), { target: { value: "30" } });
    expect(within(w).getByText(/Reaches about 13,333 people a day/)).toBeTruthy();
    expect(
      within(w).getByText(new RegExp(`${formatMoney(60_000).replace("$", "\\$")} in total`)),
    ).toBeTruthy();
    await userEvent.click(within(w).getByRole("button", { name: "Start the campaign" }));
    await waitFor(() => expect(commands()).toHaveLength(1));
    expect(commands()[0]).toEqual({
      type: "startCampaign",
      region: "pacific",
      group: "developers",
      budgetPerDay: 2000,
      days: 30,
    });
  });

  it("show a refusal in the window", async () => {
    market();
    await userEvent.click(screen.getByRole("button", { name: "New ad campaign" }));
    const w = await screen.findByRole("dialog", { name: "New ad campaign" });
    await userEvent.click(within(w).getByRole("button", { name: "Start the campaign" }));
    await waitFor(() => expect(game.ws.sentOf("command")).toHaveLength(1));
    const cid = game.ws.sentOf("command")[0]?.cid;
    game.ws.receive({ v: 1, type: "reject", cid, reason: "You need at least $500 for a day." });
    expect(await within(w).findByRole("alert")).toBeTruthy();
    expect(within(w).getByRole("alert").textContent).toBe("You need at least $500 for a day.");
  });

  it("list running campaigns with days left and spend, and stop one", async () => {
    let s = apply(
      launchedWorld(),
      command(
        "u1",
        { type: "startCampaign", region: "pacific", group: "all", budgetPerDay: 1000, days: 10 },
        "k1",
      ),
    );
    s = ticks(s, 3);
    market(viewOf(s));
    const panel = screen.getByRole("region", { name: "Ad campaigns" });
    expect(within(panel).getByText("Pacific · everyone")).toBeTruthy();
    expect(within(panel).getByText("7 days left")).toBeTruthy();
    expect(within(panel).getByText(`${formatMoney(3000)} spent`)).toBeTruthy();
    await userEvent.click(within(panel).getByRole("button", { name: "Stop the Pacific campaign" }));
    await waitFor(() => expect(commands()).toEqual([{ type: "stopCampaign", campaignId: "k1" }]));
  });

  it("open from a region's window, aimed at that region", async () => {
    market();
    const map = screen.getByRole("group", { name: "Your customers by region" });
    await userEvent.click(within(map).getByRole("button", { name: "Open Mountain West" }));
    const region = await screen.findByRole("dialog", { name: "Mountain West" });
    await userEvent.click(within(region).getByRole("button", { name: "Advertise here" }));
    const w = await screen.findByRole("dialog", { name: "New ad campaign" });
    expect((within(w).getByLabelText("Where") as HTMLSelectElement).value).toBe("mountain");
  });
});
