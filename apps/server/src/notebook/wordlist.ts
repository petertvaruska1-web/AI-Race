// Words a notebook may not hold: the strongest swear words and slurs. A notebook is
// training text, so what it holds is what the player's AI learns to say. Kept short and
// matched as whole words (with common endings), so ordinary words that contain one
// ("Scunthorpe", "class") still pass.

export const BLOCKED_WORDS: readonly string[] = [
  "fuck",
  "shit",
  "cunt",
  "bitch",
  "bastard",
  "asshole",
  "dick",
  "cock",
  "pussy",
  "whore",
  "slut",
  "fag",
  "faggot",
  "nigger",
  "nigga",
  "retard",
  "spic",
  "chink",
  "kike",
  "wetback",
  "tranny",
];
