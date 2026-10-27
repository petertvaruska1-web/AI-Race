import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
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
});

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
});
