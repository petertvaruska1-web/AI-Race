/** The HTTP side of the server: accounts and the world list. Everything else goes over the socket. */

export type Me = { userId: string; displayName: string; wsToken: string };
export type WorldSummary = {
  id: string;
  name: string;
  phase: "setup" | "running";
  day: number;
  memberCount: number;
};

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    credentials: "same-origin",
    headers: body === undefined ? {} : { "content-type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const data = (await res.json().catch(() => ({}))) as { error?: string };
  if (!res.ok)
    throw new ApiError(res.status, data.error ?? "Something went wrong. Please try again.");
  return data as T;
}

export const api = {
  guest: (displayName: string) =>
    call<{ userId: string; displayName: string }>("/api/guest", { displayName }),
  /** The signed-in player, or null when there is no session yet. */
  me: async (): Promise<Me | null> => {
    try {
      return await call<Me>("/api/me");
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) return null;
      throw err;
    }
  },
  listWorlds: () => call<{ worlds: WorldSummary[] }>("/api/worlds").then((r) => r.worlds),
  createWorld: (name: string) =>
    call<{ worldId: string }>("/api/worlds", { name }).then((r) => r.worldId),
};
