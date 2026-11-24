# Release notes: the new training text (2026-11-24)

What changed in your game, in plain words. The measurements behind every line are in `docs/decisions/0010-next-level.md`.

## What you will notice
- **A new AI talks like a small AI, not like a confused person.** Every AI now learns from training text 2: short chats with a friendly AI, follow-up questions, facts, instructions and chats in which the AI knows its name. You tried it in the second game and found it "a lot better"; that text is now the main game's.
- **It knows who it is.** The game tells each AI its name and its company when it speaks. Bigger brains usually say their name right; a first, tiny AI often says a name that only starts like it ("Sun1" for "v1").
- **New worlds start with a reading that leans on conversations:** 30% conversations and 10% each of the other seven. You can still move every slider. This made a first AI's small talk clearly better in blind tests, without losing anything else.
- **The Conversations card** now says what it holds: "Short chats between people and a friendly AI: questions, requests and small talk."
- **New rivals** for new worlds, trained on the new text (the same eight companies and styles).

## What stays the same
- **Your worlds keep their rules.** Their AIs chat and battle as before, and their rivals stay the rivals they had. An AI you train further in an old world reads the new text from its next run on, and from then on it is told its name too.
- Scores already measured stay as they were. **Creativity** is measured by a new judge from now on, so a creativity score taken after today can differ a little from one taken before.

## For playing with friends
- **Each friend runs the helper's setup file once more** before your next game together (the game's data changed; the helper says so itself).
- Their download is smaller than before: about 750 MB of game data. Rival AIs are no longer part of it; they arrive one at a time when a battle or chat needs one.

## Behind the scenes
- The old training text is kept unchanged as the measuring stick (`data/yardstick`), so later work is still measured on the same scale.
- Your worlds were backed up before anything moved (`data/server-backup-2026-11-24`).
- The second game (`data2`) is retired: its text is now the main game's. Its test worlds stay on disk.
