import { randomInt } from "node:crypto";
import { INVITE_ALPHABET, INVITE_LENGTH } from "@airace/protocol";

// The alphabet and length live in the protocol: the lab door's keypad shows the same keys.
export { INVITE_ALPHABET, INVITE_LENGTH };

export function generateInviteCode(): string {
  let code = "";
  for (let i = 0; i < INVITE_LENGTH; i++)
    code += INVITE_ALPHABET[randomInt(INVITE_ALPHABET.length)];
  return code;
}
