import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FAIR_NOTICE } from "../src/fair.ts";
import { Home, Welcome } from "../src/screens/Home.tsx";

/** Answers `fetch` by path; records every request. */
function mockFetch(routes: Record<string, unknown>) {
  const calls: { path: string; body: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string, init?: RequestInit) => {
      calls.push({ path, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      const data = routes[path];
      return new Response(JSON.stringify(data ?? { error: "not found" }), {
        status: data === undefined ? 404 : 200,
      });
    }),
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

describe("welcome", () => {
  it("asks for a name and signs the player up as a guest", async () => {
    const me = { userId: "u1", displayName: "Ana", wsToken: "t" };
    const calls = mockFetch({ "/api/guest": { userId: "u1", displayName: "Ana" }, "/api/me": me });
    const onSignedIn = vi.fn();
    render(<Welcome onSignedIn={onSignedIn} />);
    expect(screen.getByText("What should we call you?")).toBeTruthy();
    const button = screen.getByRole("button", { name: "Continue" });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Ana" } });
    fireEvent.click(button);
    await waitFor(() => expect(onSignedIn).toHaveBeenCalledWith(me));
    expect(calls[0]).toEqual({ path: "/api/guest", body: { displayName: "Ana" } });
  });

  it("prints the name on a lab ID badge as it signs in", async () => {
    const me = { userId: "u1", displayName: "Ana", wsToken: "t" };
    mockFetch({ "/api/guest": { userId: "u1", displayName: "Ana" }, "/api/me": me });
    const onSignedIn = vi.fn();
    render(<Welcome onSignedIn={onSignedIn} />);
    expect(screen.queryByTestId("id-badge")).toBeNull();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Ana" } });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(within(await screen.findByTestId("id-badge")).getByText("Ana")).toBeTruthy();
    await waitFor(() => expect(onSignedIn).toHaveBeenCalledWith(me));
  });
});

const ME = { userId: "u1", displayName: "Ana", wsToken: "t" };

describe("home", () => {
  it("lists the player's worlds and starts a new one", async () => {
    const calls = mockFetch({
      "/api/worlds": {
        worlds: [{ id: "w1", name: "Friday night", phase: "running", day: 12, memberCount: 1 }],
      },
    });
    render(<Home me={{ userId: "u1", displayName: "Ana", wsToken: "t" }} />);
    expect(await screen.findByText("Friday night")).toBeTruthy();
    expect(screen.getByText(/Day 12/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("World name"), { target: { value: "Saturday" } });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(calls.at(-1)).toEqual({ path: "/api/worlds", body: { name: "Saturday" } }),
    );
  });

  it("starts a world in Fair mode, with the notice shown first", async () => {
    const calls = mockFetch({ "/api/worlds": { worlds: [] } });
    render(<Home me={{ userId: "u1", displayName: "Ana", wsToken: "t" }} />);
    await screen.findByText(/No worlds yet/);
    expect(screen.queryByText(FAIR_NOTICE)).toBeNull();
    fireEvent.click(screen.getByLabelText("Fair mode"));
    expect(screen.getByText(FAIR_NOTICE)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("World name"), { target: { value: "Fair night" } });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(calls.at(-1)).toEqual({
        path: "/api/worlds",
        body: { name: "Fair night", fairMode: true },
      }),
    );
  });

  it("hangs each world on the rack: running, not started yet, or made with an older version", async () => {
    mockFetch({
      "/api/worlds": {
        worlds: [
          { id: "w1", name: "Friday night", phase: "running", day: 12, memberCount: 3 },
          { id: "w2", name: "Saturday", phase: "setup", day: 0, memberCount: 1 },
          { id: "w3", name: "Old times", outdated: true },
        ],
      },
    });
    render(<Home me={ME} />);
    const rack = await screen.findByRole("list", { name: "Your worlds" });
    const [running, waiting, old] = within(rack).getAllByRole("listitem") as HTMLElement[];
    expect(running?.dataset.state).toBe("running");
    expect(within(running as HTMLElement).getByText("Day 12")).toBeTruthy();
    expect(waiting?.dataset.state).toBe("setup");
    expect(within(waiting as HTMLElement).getByText("Not started yet")).toBeTruthy();
    expect(within(waiting as HTMLElement).getByRole("button", { name: "Open" })).toBeTruthy();
    expect(old?.dataset.state).toBe("outdated");
    expect(
      within(old as HTMLElement).getByText("Made with an older version of the game"),
    ).toBeTruthy();
    expect(within(old as HTMLElement).queryByRole("button", { name: "Open" })).toBeNull();
  });

  it("joins a friend's world from the keypad, a pasted code cleaned up", async () => {
    mockFetch({ "/api/worlds": { worlds: [] } });
    const openLink = vi.fn();
    render(<Home me={ME} openLink={openLink} />);
    await screen.findByText(/No worlds yet/);
    const join = screen.getByRole("button", { name: "Join" }) as HTMLButtonElement;
    expect(join.disabled).toBe(true);
    const pad = screen.getByLabelText("Invite code") as HTMLInputElement;
    fireEvent.change(pad, { target: { value: " ab c23 " } });
    expect(pad.value).toBe("ABC23");
    fireEvent.click(join);
    expect(openLink).toHaveBeenCalledWith("/join/ABC23");
  });
});
