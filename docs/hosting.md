# Playing AI Race with friends from your PC

The game runs on your PC. Friends play in their browser, and each friend's own computer can do
the AI work for their company (the "helper").

## Start the game for friends

In the AI Race folder, run:

```
npm run host
```

It builds the game, gets the helper ready for your friends, and starts the game with your own
graphics card. When it is ready it prints:

- **You play at:** `http://localhost:8787`
- **Friends play at:** a link like `https://some-words.trycloudflare.com`

Send that link to your friends. It changes every time you host, so send the new one each time.
Ctrl+C stops everything.

### The friends' link

The link comes from Cloudflare's free "quick tunnel". Install it once:

```
winget install Cloudflare.cloudflared
```

Without it, only friends on your own network (the same Wi-Fi) can join, at the address the game
prints ("Friends on your network play at …"). Windows may ask whether to let Node.js through the
firewall: allow it on private networks.

## What a friend does

1. Opens your link and picks a name (no sign-up).
2. Joins your world with its invite code (the world's settings show it).
3. Presses **Connect your computer** → **Get the helper**, and runs the downloaded
   `AI Race Helper setup.cmd`. The first time takes about 15 minutes: it installs Python and the
   AI libraries (about 3 GB) and downloads the game's data (about 1.1 GB). Later it only fetches
   what changed.
4. Presses **Connect your computer** again whenever they play: the browser asks to open the
   helper, which then trains their AIs on their own graphics card.

Without a graphics card the helper uses the processor: it works, slowly.

## Fair mode

Chosen when a world is made, and fixed for that world:

- **Off:** every company's AI is trained and run on its own player's computer. A faster card
  trains faster.
- **On:** all the players' cards are shared, and each company gets an equal share. A run can move
  to another player's card if its card goes away. Because work moves between PCs, an AI's files
  may pass through another player's computer: the game says so when you choose it.

## Privacy

Friends only ever see what the game shows them. Their setup file holds their own helper key:
it lets their helper work for them in your games. If a key leaks, the game can make a new one.

## For checks (technical)

`AIRACE_DATA`, `AIRACE_SERVER_PORT`, `AIRACE_NO_TUNNEL=1` and `HOST=127.0.0.1` run a second,
private copy beside a game in progress, e.g. on the `data-check` root:

```
AIRACE_DATA=data-check AIRACE_SERVER_PORT=8790 AIRACE_NO_TUNNEL=1 HOST=127.0.0.1 npm run host
```
