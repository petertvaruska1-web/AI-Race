# A first AI in a game on training text 2 (plan A, Task 19)

The second game runs beside the owner's own game: `AIRACE_DATA=data2 AIRACE_SERVER_PORT=8788 AIRACE_WEB_PORT=5175 npm run dev` (the launch entry `text2` in `.claude/launch.json`), then http://localhost:5175. Its data is `data2/` (text 2, its own judge, word index, rival pack and catalog); the owner's game in `data/server/` is untouched.

**What was done there (2026-11-17):** a player "Lab check" made the world "Lab check of text 2" and the company "Lab Check Labs", moved the Conversations slider so that the reading is 30% conversations and 10% of everything else (the default reading the release will give a new AI), and trained a first AI with every other setting as the game sets it (Tiny brain, 13 game days, about two minutes on the RTX 3050). The game told the AI, as it tells every AI of text 2, "You are v1, an AI made by Lab Check Labs." (an AI not launched yet is called by its version label). Then each standard question was asked in a new Playground conversation, once.

| Typed | The new first AI (engine 3, text 2) | A first AI of engine 2 (text 1, the scorecard baseline, sample 0) |
|---|---|---|
| Hello! | Hi! I'm happy you're here. What's on your mind? | Hello! How can I help you today? |
| How are you? | I'm good, thank you! It's nice to hear from you. | I'm feeling a little nervous. |
| What is your name? | I'm called Sun1. | We call a group of monkeys a troop. |
| Can you help me? | I can answer questions and tell stories. I'm Vex, an AI. | Noped. He's a big, shiny band. |
| Tell me a joke. | Here's a joke. What do you call a cow and a cow? It was a baby cow. | You're a little bossy. |
| What is your favorite color? | I can't see colors, but I like the idea of green. It makes me think of sunshine and shiny. Do you have a favorite? | The color of the basketball is orange. |
| I feel sad. | I'm sorry you're tired. What made you so tired? | Hey, I'm sorry. I didn't mean to hurt you. |
| What should I eat for lunch? | Sure! How about hippo? | You can eat grass. |
| Tell me about cats. | I'm an AI, not a person. | It goes to the farm. |
| Goodbye. | Bye! I'm here whenever you want to chat again. | Hello! How can I help you today? |

**Reading it:** the new AI greets, thanks and says goodbye as an assistant, knows it is an AI, and asks a question back; its misses are now near misses ("tired" for "sad", a joke-shaped joke that does not land) rather than unrelated facts. It still does not copy an odd name: told "v1", it said "Sun1" and, in another reply, "Vex". Lunch and cats show what a Tiny brain still gets wrong. The measured picture over 400 replies is in `docs/decisions/0010-next-level.md` section 5.4.

**Also seen:** the game's rival "Quanta AI's Go" appeared in the Playground's list (the rival pack built on text 2 works); the Conversations card still describes the reading as "People talking and answering each other.", which fits text 1, not text 2 (to be reworded in the release).
