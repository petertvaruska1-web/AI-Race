import type { CompanyState, VersionState, WorldState } from "@airace/sim";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Directory } from "../src/screens/Directory.tsx";
import { VersionsStep } from "../src/screens/lab/VersionsStep.tsx";
import { Playground } from "../src/screens/Playground.tsx";
import { useCards } from "../src/store/cards.ts";
import { usePlayground } from "../src/store/playground.ts";
import { useSession } from "../src/store/session.ts";
import { liveGame } from "./mockSocket.ts";
import { apply, command, LOGO, PALETTE, runningWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  usePlayground.setState({ versionId: null });
  location.hash = "";
});
afterEach(() => useCards.setState({ worldId: null, cards: null, limit: null }));

function version(id: string, label: string, published: boolean): VersionState {
  return {
    id,
    label,
    runId: `r-${id}`,
    parentId: null,
    nLayer: 2,
    dModel: 64,
    ctxLen: 64,
    createdDay: 0,
    trainedTokens: 1,
    scores: null,
    overall: 42,
    personalityWords: ["curious"],
    traits: null,
    published,
    packKey: null,
  } as VersionState;
}

/** Ana (u1, c1) with her v1; Ben (u2, c2, "Bolt") sharing his Nova 1. */
function withFriend(): WorldState {
  let s = runningWorld("balance-v2");
  s = apply(s, {
    kind: "system",
    event: { type: "memberJoined", userId: "u2", displayName: "Ben" },
  });
  s = apply(
    s,
    command("u2", { type: "createCompany", name: "Bolt", logo: LOGO, palette: PALETTE }, "c2"),
  );
  (s.companies.c1 as CompanyState).versions.v1 = version("v1", "Spark 1", false);
  (s.companies.c2 as CompanyState).versions.w1 = version("w1", "Nova 1", true);
  return s;
}

describe("the AI Directory", () => {
  it("lists a friend's shared AI, and Try it opens the Playground on it", () => {
    render(<Directory view={viewOf(withFriend())} />);
    const row = within(screen.getByText("Nova 1").closest("li") as HTMLElement);
    expect(row.getByText("Bolt")).toBeTruthy();
    expect(row.getByText(/curious/)).toBeTruthy();
    expect(screen.queryByText("Spark 1")).toBeNull(); // your own, unshared
    fireEvent.click(row.getByRole("button", { name: "Try it" }));
    expect(usePlayground.getState().versionId).toBe("w1");
    expect(location.hash).toContain("/play");
  });

  it("can't try an AI whose maker's computer is off (Fair mode off)", () => {
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
    render(<Directory view={viewOf(withFriend())} />);
    const row = within(screen.getByText("Nova 1").closest("li") as HTMLElement);
    expect(row.getByText("Its maker's computer is offline")).toBeTruthy();
    expect((row.getByRole("button", { name: "Try it" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("a friend's AI in the Playground says whose computer answers", () => {
    usePlayground.setState({ versionId: "w1" });
    render(<Playground view={viewOf(withFriend())} />);
    expect(screen.getByText("Its maker's computer answers your messages.")).toBeTruthy();
  });
});

describe("sharing with friends", () => {
  it("shares a version and shows how often friends tried it", async () => {
    let s = withFriend();
    (s.companies.c1 as CompanyState).versions.v1 = version("v1", "Spark 1", true);
    s = { ...s };
    render(<VersionsStep view={viewOf(s)} onTrainFurther={() => {}} />);
    const q = await waitFor(() => {
      const sent = game.ws.sentOf("query").find((m) => m.q.kind === "versionTries");
      if (!sent) throw new Error("no query yet");
      return sent;
    });
    game.ws.receive({
      v: 1,
      type: "queryResult",
      id: q.id,
      ok: true,
      data: { versions: [{ versionId: "v1", tries7d: 3 }] },
    });
    expect(await screen.findByText("Tried 3 times this week")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Stop sharing" }));
    await waitFor(() =>
      expect(game.ws.sentOf("command").at(-1)?.cmd).toEqual({
        type: "publishVersion",
        versionId: "v1",
        published: false,
      }),
    );
  });

  it("sends publishVersion to share", async () => {
    render(<VersionsStep view={viewOf(withFriend())} onTrainFurther={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Share with friends" }));
    await waitFor(() =>
      expect(game.ws.sentOf("command").at(-1)?.cmd).toEqual({
        type: "publishVersion",
        versionId: "v1",
        published: true,
      }),
    );
  });
});
