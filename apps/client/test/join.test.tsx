import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FAIR_NOTICE } from "../src/fair.ts";
import { joinCodeOf } from "../src/route.ts";
import { Join } from "../src/screens/Join.tsx";

/** Answers `fetch` by path with [status, body]; records every request. */
function mockFetch(routes: Record<string, [number, unknown] | (() => [number, unknown])>) {
  const calls: { path: string; body: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string, init?: RequestInit) => {
      calls.push({ path, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      const route = routes[path];
      const [status, data] = typeof route === "function" ? route() : (route ?? [404, {}]);
      return new Response(JSON.stringify(data), { status });
    }),
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

const ME = { userId: "u2", displayName: "Ben", wsToken: "t" };
const FAIR_WORLD = {
  name: "Friday night",
  fairMode: true,
  players: 2,
  maxPlayers: 8,
  member: false,
};

describe("the join link", () => {
  it("is read from the address", () => {
    expect(joinCodeOf("/join/abc123")).toBe("ABC123");
    expect(joinCodeOf("/join/ABC123/")).toBe("ABC123");
    expect(joinCodeOf("/")).toBeNull();
  });

  it("asks a new player for a name, shows the world and its Fair mode notice, then joins", async () => {
    let signedIn = false;
    const calls = mockFetch({
      "/api/guest": () => {
        signedIn = true;
        return [200, { userId: "u2", displayName: "Ben" }];
      },
      "/api/me": () => (signedIn ? [200, ME] : [401, { error: "Please sign in again." }]),
      "/api/worlds/invite/ABC123": [200, FAIR_WORLD],
      "/api/worlds/join": [200, { worldId: "w1" }],
    });
    const onJoined = vi.fn();
    const onSignedIn = vi.fn();
    render(<Join code="ABC123" me={null} onSignedIn={onSignedIn} onJoined={onJoined} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Ben" } });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(onSignedIn).toHaveBeenCalledWith(ME));
    expect(calls[0]).toEqual({ path: "/api/guest", body: { displayName: "Ben" } });
  });

  it("for a signed-in player shows the world and the Fair mode notice, then joins", async () => {
    const calls = mockFetch({
      "/api/worlds/invite/ABC123": [200, FAIR_WORLD],
      "/api/worlds/join": [200, { worldId: "w1" }],
    });
    const onJoined = vi.fn();
    render(<Join code="ABC123" me={ME} onSignedIn={vi.fn()} onJoined={onJoined} />);
    expect(await screen.findByText("Friday night")).toBeTruthy();
    expect(screen.getByText(/2 of 8 players/)).toBeTruthy();
    expect(screen.getByText(FAIR_NOTICE)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Back" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Join" }));
    await waitFor(() => expect(onJoined).toHaveBeenCalledWith("w1"));
    expect(calls.at(-1)).toEqual({ path: "/api/worlds/join", body: { code: "ABC123" } });
  });

  it("shows why a world can't be joined", async () => {
    mockFetch({
      "/api/worlds/invite/ABC123": [200, { ...FAIR_WORLD, fairMode: false }],
      "/api/worlds/join": [409, { error: "This world is full (8 players)." }],
    });
    const onJoined = vi.fn();
    render(<Join code="ABC123" me={ME} onSignedIn={vi.fn()} onJoined={onJoined} />);
    expect(await screen.findByText("Friday night")).toBeTruthy();
    expect(screen.queryByText(FAIR_NOTICE)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Join" }));
    expect(await screen.findByText("This world is full (8 players).")).toBeTruthy();
    expect(onJoined).not.toHaveBeenCalled();
  });

  it("says when a code matches no world", async () => {
    mockFetch({
      "/api/worlds/invite/ZZZZZZ": [404, { error: "That invite code doesn't match a world." }],
    });
    render(<Join code="ZZZZZZ" me={ME} onSignedIn={vi.fn()} onJoined={vi.fn()} />);
    expect(await screen.findByText("That invite code doesn't match a world.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Join" })).toBeNull();
  });
});
