import type { Identity } from "@airace/protocol";
import type { CompanyState, WorldState } from "@airace/sim";
import type { AppContext } from "../app.ts";

/** The longest name or company the worker protocol takes (`Identity` in the protocol). */
const MAX_NAME = 60;

function cut(text: string): string {
  const chars = [...text.trim()];
  return chars.length > MAX_NAME ? chars.slice(0, MAX_NAME).join("").trimEnd() : chars.join("");
}

/**
 * What an AI is told about itself: its public name and its company's public name, both of which
 * every player already sees in their view (R2). No name, no identity: never an empty string.
 */
export function makeIdentity(
  name: string | null | undefined,
  company: string | null | undefined,
): Identity | undefined {
  const aiName = cut(name ?? "");
  if (!aiName) return undefined;
  const maker = cut(company ?? "");
  return { name: aiName, company: maker || null };
}

/**
 * The AI behind version `versionId`, as the players know it: the name of the product it powers
 * (on sale first, then a launch that is counting down), else the version's own label. A retired
 * product's name is not in anybody's view, so it is not used.
 */
export function identityOf(s: WorldState, versionId: string): Identity | undefined {
  for (const company of Object.values(s.companies)) {
    const version = company.versions[versionId];
    if (version) return makeIdentity(publicNameOf(company, versionId, version.label), company.name);
  }
  return undefined;
}

/** A rival AI by its pack key (the key the worker's model is named by). */
export function identityOfPack(s: WorldState, packKey: string): Identity | undefined {
  for (const company of Object.values(s.companies)) {
    for (const version of Object.values(company.versions)) {
      if (version.packKey === packKey) {
        return makeIdentity(publicNameOf(company, version.id, version.label), company.name);
      }
    }
  }
  return undefined;
}

function publicNameOf(company: CompanyState, versionId: string, label: string): string {
  const powered = Object.values(company.products).filter(
    (p) => p.versionId === versionId && p.status !== "retired",
  );
  const product = powered.find((p) => p.status === "live") ?? powered[0];
  return product?.name ?? label;
}

/**
 * The identity of version `versionId` in world `worldId`, or none when the world can't be read:
 * a job still runs without it (the helper then adds no line).
 */
export async function identityIn(
  ctx: AppContext,
  worldId: string,
  versionId: string,
): Promise<Identity | undefined> {
  try {
    const actor = ctx.worlds.peek(worldId) ?? (await ctx.worlds.get(worldId));
    return identityOf(actor.state(), versionId);
  } catch {
    return undefined;
  }
}
