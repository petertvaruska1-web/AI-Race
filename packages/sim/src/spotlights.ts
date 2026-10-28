import { GROUP_IDS, type GroupId, REGION_IDS, type RegionId } from "@airace/protocol";
import type { Economy } from "./balance.ts";
import { type RngState, rngInt } from "./rng.ts";
import type { Spotlight } from "./state.ts";

// Six named customers per world, one per group, who put a face on the numbers.

const FIRST_NAMES = [
  "Maya",
  "Leo",
  "Priya",
  "Sam",
  "Ava",
  "Noah",
  "Zoe",
  "Omar",
  "Lena",
  "Jay",
  "Iris",
  "Theo",
  "Nina",
  "Ravi",
  "Cleo",
  "Max",
  "Hana",
  "Eli",
  "Rosa",
  "Kai",
  "Jade",
  "Finn",
  "Lucia",
  "Ezra",
];
const LAST_INITIALS = "ABCDEFGHJKLMNPRSTVW";
const BIOS: Record<GroupId, string[]> = {
  students: [
    "Studies biology and never has enough time.",
    "First-year student cramming for exams.",
    "Writes essays at 2 a.m.",
  ],
  creators: [
    "Writes short stories for an online audience.",
    "Makes videos and needs fresh ideas.",
    "Draws comics and writes their scripts.",
  ],
  developers: [
    "Builds web apps at a small startup.",
    "Maintains an open-source library.",
    "Teaches herself to code at night.",
  ],
  businesses: [
    "Runs a twelve-person accounting firm.",
    "Manages support at an online shop.",
    "Owns a busy family restaurant.",
  ],
  everyday: [
    "Likes quizzes and planning trips.",
    "Retired teacher, curious about everything.",
    "Busy parent of three.",
  ],
  researchers: [
    "Studies climate data at a university.",
    "Analyses medical trials.",
    "Works on puzzles in a maths lab.",
  ],
};

/** A customer who is not a spotlight: a name and handle drawn from the world rng. */
export function drawAuthor(rng: RngState): { name: string; handle: string; rng: RngState } {
  let r = rng;
  const draw = (n: number) => {
    const x = rngInt(r, n);
    r = x.next;
    return x.value;
  };
  const first = FIRST_NAMES[draw(FIRST_NAMES.length)] ?? "Sam";
  const initial = LAST_INITIALS[draw(LAST_INITIALS.length)] ?? "A";
  return {
    name: `${first} ${initial}.`,
    handle: `@${first.toLowerCase()}_${initial.toLowerCase()}${10 + draw(90)}`,
    rng: r,
  };
}

/** Draws the world's six spotlight customers; each lives in a region drawn by its group's people. */
export function drawSpotlights(
  rng: RngState,
  economy: Economy,
): { spotlights: Spotlight[]; rng: RngState } {
  let r = rng;
  const draw = (n: number) => {
    const x = rngInt(r, n);
    r = x.next;
    return x.value;
  };
  const used = new Set<string>();
  const spotlights: Spotlight[] = GROUP_IDS.map((group, i) => {
    let first = FIRST_NAMES[draw(FIRST_NAMES.length)] ?? "Sam";
    while (used.has(first)) first = FIRST_NAMES[draw(FIRST_NAMES.length)] ?? "Sam";
    used.add(first);
    const initial = LAST_INITIALS[draw(LAST_INITIALS.length)] ?? "A";
    const bios = BIOS[group];
    const weights = REGION_IDS.map((r) => economy.population[r] * economy.groupMix[r][group]);
    let pick = (draw(1_000_000) / 1_000_000) * weights.reduce((a, b) => a + b, 0);
    let region: RegionId = REGION_IDS[REGION_IDS.length - 1] ?? "pacific";
    for (const [k, w] of weights.entries()) {
      pick -= w;
      if (pick < 0) {
        region = REGION_IDS[k] ?? region;
        break;
      }
    }
    return {
      id: `spot-${i + 1}`,
      name: `${first} ${initial}.`,
      handle: `@${first.toLowerCase()}_${initial.toLowerCase()}${10 + draw(90)}`,
      group,
      region,
      bio: bios[draw(bios.length)] ?? "",
    };
  });
  return { spotlights, rng: r };
}
