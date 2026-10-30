import type { Command, WorldView } from "@airace/protocol";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { Arena } from "../src/screens/Arena.tsx";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { apply, command, rivalWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;
beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useWindows.getState().closeAll();
});

const commands = () => game.ws.sentOf("command").map((m) => m.cmd as Command);

function arena(view: WorldView) {
  return render(
    <>
      <Arena view={view} />
      <WindowHost view={view} />
    </>,
  );
}

function archTarget(view: WorldView) {
  const arch = view.companies.find((c) => c.rival?.archetype);
  const live = arch?.products.find((p) => p.status === "live");
  if (!arch || !live) throw new Error("the arch-rival has nothing on sale");
  return { arch, live };
}

/** The world with battle b1 against the arch-rival started, and optionally resolved 2–1. */
function battleWorld(resolved: boolean) {
  const base = rivalWorld();
  const { live } = archTarget(viewOf(base));
  let s = apply(
    base,
    command("u1", { type: "startBattle", rivalProductId: live.id, skill: "knowledge" }, "b1"),
  );
  if (resolved) {
    s = apply(s, {
      kind: "system",
      event: {
        type: "battleResolved",
        battleId: "b1",
        rounds: [0, 1, 2].map((k) => ({
          prompt: `Question ${k}?`,
          mine: { reply: k === 0 ? "<b>x</b> <|end|>" : "yes", score: k < 2 ? 1 : 0 },
          theirs: { reply: "no", score: k < 2 ? 0 : 1 },
        })),
      },
    });
  }
  return viewOf(s);
}

describe("the Arena", () => {
  it("starts a battle with exactly the rival AI and skill chosen", async () => {
    const view = viewOf(rivalWorld());
    const { live } = archTarget(view);
    arena(view);
    await userEvent.click(screen.getByRole("button", { name: "Start a battle" }));
    const w = await screen.findByRole("dialog", { name: "Start a battle" });
    await userEvent.selectOptions(within(w).getByLabelText("Against"), live.id);
    await userEvent.click(within(w).getByRole("radio", { name: /Knowledge/ }));
    expect(within(w).getByText(/\$5,000/)).toBeTruthy();
    await userEvent.click(within(w).getByRole("button", { name: "Start the battle" }));
    await waitFor(() =>
      expect(commands()).toEqual([
        { type: "startBattle", rivalProductId: live.id, skill: "knowledge" },
      ]),
    );
    const cid = game.ws.sentOf("command")[0]?.cid;
    game.ws.receive({ v: 1, type: "reject", cid, reason: "Your last battle is still going." });
    expect((await within(w).findByRole("alert")).textContent).toBe(
      "Your last battle is still going.",
    );
  });

  it("streams both replies of a running battle into their columns, as text", async () => {
    const view = battleWorld(false);
    arena(view);
    await userEvent.click(screen.getByRole("button", { name: /Open the knowledge battle/ }));
    const w = await screen.findByRole("dialog", { name: /battle/ });
    await waitFor(() => expect(game.ws.sentOf("query").length).toBeGreaterThan(0));
    const q = game.ws.sentOf("query").at(-1);
    game.ws.receive({ v: 1, type: "queryResult", id: q?.id, ok: true, data: { rounds: [] } });
    await waitFor(() =>
      expect(game.ws.sentOf("subscribe").map((m) => m.sub)).toContain("battle:w1:b1"),
    );
    const stream = (data: unknown) =>
      game.ws.receive({ v: 1, type: "stream", sub: "battle:w1:b1", data });
    stream({ kind: "text", round: 0, side: "mine", text: "<i>Par" });
    stream({ kind: "text", round: 0, side: "theirs", text: "Lon" });
    expect(await within(w).findByText("<i>Par")).toBeTruthy();
    expect(within(w).getByTestId("round-0-mine").textContent).toBe("<i>Par");
    expect(within(w).getByTestId("round-0-theirs").textContent).toBe("Lon");
  });

  it("shows a finished battle's stamp, replies and ranking points from the view", async () => {
    const view = battleWorld(true);
    arena(view);
    await userEvent.click(screen.getByRole("button", { name: /Open the knowledge battle/ }));
    const w = await screen.findByRole("dialog", { name: /battle/ });
    expect(within(w).getByText("Win 2–1")).toBeTruthy();
    expect(within(w).getByText("<b>x</b> <|end|>")).toBeTruthy();
    expect(within(w).getByText("+16 ranking points")).toBeTruthy();
  });

  it("ranks every company and hides other companies' revenue", async () => {
    const view = battleWorld(true);
    arena(view);
    await userEvent.click(screen.getByRole("tab", { name: "Revenue" }));
    const table = screen.getByRole("table", { name: "Revenue" });
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(view.companies.length);
    for (const row of rows) {
      const mine = row.getAttribute("data-mine") === "true";
      const cell = within(row).getAllByRole("cell").at(-1);
      if (!mine) expect(cell?.textContent).toBe("—");
    }
    await userEvent.click(screen.getByRole("tab", { name: "Ranking points" }));
    expect(screen.getByRole("table", { name: "Ranking points: knowledge" })).toBeTruthy();
  });
});
