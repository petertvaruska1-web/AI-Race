import type { CompanyState, ProductState, VersionState, WorldState } from "@airace/sim";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Arena } from "../src/screens/Arena.tsx";
import { useCards } from "../src/store/cards.ts";
import { useSession } from "../src/store/session.ts";
import { useVotes } from "../src/store/votes.ts";
import { useWindows } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { apply, command, LOGO, PALETTE, runningWorld, SCORES, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useWindows.setState({ stack: [], seq: 0 });
});
afterEach(() => {
  useCards.setState({ worldId: null, cards: null, limit: null });
  useVotes.setState({ worldId: null, ballots: {} });
});

function version(id: string): VersionState {
  return {
    id,
    label: `${id} 1`,
    runId: `r-${id}`,
    parentId: null,
    nLayer: 2,
    dModel: 64,
    ctxLen: 64,
    createdDay: 0,
    trainedTokens: 1,
    scores: { ...SCORES },
    overall: 40,
    personalityWords: [],
    traits: null,
    published: false,
    packKey: null,
  } as VersionState;
}

function onSale(c: CompanyState, versionId: string, productId: string, name: string) {
  c.versions[versionId] = version(versionId);
  c.products[productId] = {
    id: productId,
    versionId,
    name,
    status: "live",
    plans: ["free"],
    launchDay: 0,
  } as unknown as ProductState;
}

/** Ana (u1, c1) sells Spark; Ben (u2, c2 "Bolt") sells Nova. */
function friends(): WorldState {
  let s = runningWorld("balance-test");
  s = apply(s, {
    kind: "system",
    event: { type: "memberJoined", userId: "u2", displayName: "Ben" },
  });
  s = apply(
    s,
    command("u2", { type: "createCompany", name: "Bolt", logo: LOGO, palette: PALETTE }, "c2"),
  );
  onSale(s.companies.c1 as CompanyState, "v1", "p1", "Spark");
  onSale(s.companies.c2 as CompanyState, "w1", "q1", "Nova");
  return s;
}

describe("battling a friend's AI", () => {
  it("the picker lists players' AIs and starts a battle with their product", async () => {
    const view = viewOf(friends());
    useWindows.getState().open({ kind: "newBattle" });
    render(<WindowHost view={view} />);
    const against = screen.getByRole("combobox") as HTMLSelectElement;
    expect([...against.options].map((o) => o.textContent)).toContain("Bolt's Nova");
    fireEvent.change(against, { target: { value: "q1" } });
    fireEvent.click(screen.getByRole("button", { name: /Start the battle/ }));
    await waitFor(() =>
      expect(game.ws.sentOf("command").at(-1)?.cmd).toEqual({
        type: "startBattle",
        rivalProductId: "q1",
        skill: "knowledge",
      }),
    );
  });

  it("a friend whose computer is off can't be picked (Fair mode off)", () => {
    useCards.setState({
      worldId: "w",
      cards: [
        {
          userId: "u2",
          playerName: "Ben",
          gpuName: null,
          vramMb: 0,
          speed: 0,
          online: false,
          house: false,
        },
      ],
      limit: null,
    });
    useWindows.getState().open({ kind: "newBattle" });
    render(<WindowHost view={viewOf(friends())} />);
    const option = [...(screen.getByRole("combobox") as HTMLSelectElement).options].find(
      (o) => o.value === "q1",
    );
    expect(option?.disabled).toBe(true);
    expect(option?.textContent).toContain("computer off");
  });

  it("a battle you defended shows in your Arena, your side first", () => {
    const s = friends();
    (s.companies.c2 as CompanyState).battles.push({
      id: "b1",
      day: 1,
      skill: "knowledge",
      rivalId: "c1",
      mine: { productId: "q1", versionId: "w1", productName: "Nova" },
      theirs: { productId: "p1", versionId: "v1", productName: "Spark", packKey: "" },
      seed: 1,
      status: "done",
      rounds: [],
      outcome: "win",
      pointsChange: 12,
      opponentKind: "player",
      defending: true,
    });
    render(<Arena view={viewOf(s, "u2")} />);
    const item = screen.getByRole("button", { name: /knowledge battle against Acme/ });
    expect(item.textContent).toContain("Acme challenged you");
    expect(item.textContent).toContain("win");
  });
});

describe("voting on a story battle", () => {
  it("shows both stories unlabelled, sends the pick, and follows the vote stream", async () => {
    const view = viewOf(friends(), "u2");
    const stop = useVotes.getState().follow(game.socket, view.world.id);
    game.ws.receive({
      v: 1,
      type: "stream",
      sub: `battlevote:${view.world.id}`,
      data: {
        kind: "open",
        battleId: "b9",
        closesInMs: 300_000,
        rounds: [
          { round: 0, prompt: "Write a story", left: "Once upon a time", right: "A dragon" },
        ],
      },
    });
    await waitFor(() => expect(useVotes.getState().ballots.b9).toBeTruthy());
    useWindows.getState().open({ kind: "vote", battleId: "b9" });
    render(<WindowHost view={view} />);
    const round = within(screen.getByTestId("vote-round-0"));
    expect(round.getByText("Once upon a time")).toBeTruthy();
    expect(round.getByText("A dragon")).toBeTruthy();
    expect(screen.getByTestId("vote-round-0").textContent).not.toMatch(/Acme|Bolt|Spark|Nova/);
    expect(screen.getByText(/left to vote/)).toBeTruthy();
    fireEvent.click(round.getAllByRole("button", { name: "This one" })[1] as HTMLElement);
    await waitFor(() =>
      expect(game.ws.sentOf("battleVote").at(-1)).toMatchObject({
        battleId: "b9",
        round: 0,
        pick: "right",
        worldId: view.world.id,
      }),
    );
    game.ws.receive({
      v: 1,
      type: "stream",
      sub: `battlevote:${view.world.id}`,
      data: { kind: "closed", battleId: "b9", rounds: [{ round: 0, left: 1, right: 2 }] },
    });
    await waitFor(() =>
      expect(useVotes.getState().ballots.b9?.closed).toEqual([{ round: 0, left: 1, right: 2 }]),
    );
    stop();
  });

  it("a new vote pops up as a toast that opens the vote", async () => {
    const { Toasts } = await import("../src/components/Toasts.tsx");
    const view = viewOf(friends(), "u2");
    render(<Toasts view={view} />);
    useVotes.setState({
      worldId: view.world.id,
      ballots: {
        b9: {
          battleId: "b9",
          rounds: [{ round: 0, prompt: "Write a story", left: "A", right: "B" }],
          closesAt: Date.now() + 300_000,
          picks: {},
          closed: null,
        },
      },
    });
    const toast = await screen.findByText("Vote on a story battle");
    fireEvent.click(toast);
    expect(useWindows.getState().stack.at(-1)?.spec).toEqual({ kind: "vote", battleId: "b9" });
  });
});
