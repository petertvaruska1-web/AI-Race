import type { Command, WorldView } from "@airace/protocol";
import { createWorld, type Input, projectView, step, type WorldState } from "@airace/sim";

// Real world views, made by the sim, for the client tests.

export const LOGO = { shape: "circle", symbol: "spark", initials: "AC" } as const;
export const PALETTE = { primary: "#4f8cff", secondary: "#ffb020" };

export function testWorld(): WorldState {
  return createWorld({
    worldId: "w1",
    name: "Friday night",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 1,
    balance: "balance-test",
    inviteCode: "ABC234",
  });
}

export function apply(s: WorldState, input: Input): WorldState {
  const r = step(s, input);
  if (!r.ok) throw new Error(r.reason);
  return r.state;
}

export const command = (userId: string, cmd: unknown, newId?: string): Input => ({
  kind: "command",
  userId,
  cmd: cmd as Command,
  ...(newId ? { newId } : {}),
});

/** A running world where u1 owns company c1. */
export function runningWorld(): WorldState {
  let s = testWorld();
  s = apply(
    s,
    command("u1", { type: "createCompany", name: "Acme", logo: LOGO, palette: PALETTE }, "c1"),
  );
  return apply(s, command("u1", { type: "startWorld" }));
}

export function viewOf(s: WorldState, userId = "u1"): WorldView {
  return projectView(s, userId);
}
