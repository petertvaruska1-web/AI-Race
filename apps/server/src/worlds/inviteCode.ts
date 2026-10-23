import { randomInt } from "node:crypto";

// No 0/O, 1/I/L: codes are read aloud and typed by friends.
export const INVITE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";
export const INVITE_LENGTH = 6;

export function generateInviteCode(): string {
  let code = "";
  for (let i = 0; i < INVITE_LENGTH; i++)
    code += INVITE_ALPHABET[randomInt(INVITE_ALPHABET.length)];
  return code;
}
