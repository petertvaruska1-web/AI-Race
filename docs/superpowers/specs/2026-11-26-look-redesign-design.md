# The look redesign ("Instrument bench") — design

**Date:** 2026-11-26 · **Owner decisions:** C57–C66 (§0) · **Applies to:** the web client `apps/client` (every screen, window and object), plus the small protocol, sim and server changes it needs (§13) · **Before:** the ultimate game document (`docs/handoff/next-steps.md`), whose screenshots must show this look.

## 0. Owner decisions (2026-11-26)

The owner's brief, in their words where possible:

- **B1 Overall:** professional, unique and visually appealing.
- **B2 The beginning** (choosing a name, logo and so on) is a visual, interactive experience with sliders, gears and other things that make you feel you are in an AI lab, with a much more customizable logo.
- **B3 No webpage feel:** no button or page should feel like a webpage you scroll through and click buttons on. The layout feels intentional, interactive and stimulating.
- **B4 The background** is light or dark "pixelated" glass in the player's color, like looking through glass into a lab room where an experiment is going on but nothing is on the other side, with aurora-like smooth flowing waves of the player's other color.
- **B5 Real 3D objects:** every object in the game (the AI's geometric shape, the US map and so on) is a real 3D model that looks intentional, not a shape drawn by the browser. The AI's shape glows and looks more real. The US map can be rotated and its perspective changed, like an artifact in a video game; each region glows the color of the company with the most users there, and the map looks the same for everybody.
- **B6 AI challenges** are too quick and chaotic, usually over in seconds before the player realizes what is going on: slow them down and make them enjoyable.
- **B7 The final step:** use every available tool to find generic, AI-made-looking design and redo it.

Answered in the interview (multiple-choice questions and three live mockups, all in `.superpowers/brainstorm/754-1795680674/content/`):

| # | Decision |
|---|---|
| C57 | **The glass follows the main color.** Deep main colors get light glass, bright main colors get dark glass, decided automatically (no player choice). |
| C58 | **Logo = an emblem maker with dials and sliders** (frame, symbol or letters, rings, patterns; corners, turn, thickness, glow, finish). No free drawing and no uploads, because every player sees every logo. |
| C59 | **AI challenges take about a minute to watch**, with a "faster" button and "skip to the result" always there. Applies to Playground games, Arena battles and press showcases. |
| C60 | **The look is "B · Instrument bench"** (picked from three live looks after comparing A and B many times): machined metal frames with screws, engraved labels, knobs, switches and push buttons with lamps, recessed dark displays with glowing dot-matrix readouts, the AI inside geared metal rings, a station knob that turns when you change screens. Reference: `directions.html`, look B. |
| C61 | **Founding happens on one bench, exactly as in the mockup** (`founding.html`): a dot-matrix name plate, two color dials with brightness sliders and twelve quick-color cartridges, the emblem press with the 3D badge, the knobs, symbol keys, sliders and switches, the players and invite modules, and a lever you pull down to found the company. No guide light and no step-by-step. |
| C62 | **The lab door:** you sign in by typing your name onto a lab ID badge that prints; your worlds hang as lit panels on a wall rack; a new world is set up on a small bench (name plate, goal knob, Fair mode switch); joining a friend uses a keypad for the code. |
| C63 | **New pictures in the lab style replace every emoji** (about 90: the Story Dice faces, the pattern puzzles' animals, the trophy marks, the game and sponsor marks), in one consistent style: engraved on the 3D dice and trophies, lit on the displays. |

Added by the owner on 2026-11-28, while plan 4 was being built ("keep this in mind and have it fixed by the time the redesign is done"):

| # | Decision |
|---|---|
| C64 | **Benchmarks become one global graph:** everyone's AIs recorded and compared at the same time, and the player toggles what they see: hide a company's AIs, or look at one category only (coding, for example). It replaces the per-version gauges of §10. |
| C65 | **The cosmetics get a proper redesign and look much cooler** (the Customize window: orb styles and logo motions, what you buy and how you try it on). |
| C66 | **The investment fund gets a graph** of the portfolio growing or shrinking. |

Claude's reading of C64 (the owner can veto it): the graph's line for each company runs through its AIs by the day each was made, at each AI's measured score (overall, or the chosen category). Other companies' AIs appear once they are released or shared, and stay on the graph after they are retired (a public record); your own AIs all appear, released or not. Unreleased AIs of others never appear (R2).

Earlier decisions that still bind: **C5** (the company palette themes the player's screens; now it themes the glass, the aurora, the lamps and the displays), **C7** (a playful tycoon with a clean, smart, futuristic feel), **C30** (immersion first: motion, windows that open from what you clicked, things to click), **C46** (the tutorial orb guide, when it is built, is the AI core of §8.1), **C47** (music and sound: the owner leads; this redesign adds no sound).

Assumptions made here (the owner can veto them while reading):

- **A1** Engraved hardware labels are in capitals (the only place capitals are used); everything else is sentence case.
- **A2** On the Company bench, the smaller panels (team, licenses, partner apps, goal, trophies) share one "drawer bay" with a selector, one shown at a time, instead of a long column.
- **A3** Other companies' colors are shown adjusted for the viewer's glass (§5.3), so a deep color stays readable on a friend's dark glass.
- **A4** The tiny logos in tables and lists are flat renders of the same emblem; the 3D badge appears from about 64 pixels up.

## 1. Intent

The owner wants the game to look like nothing else before the game document is written, because the document's screenshots will be how other people and AI agents first meet the game. Today every screen is a dark navy web page of rounded cards (`apps/client/src/theme/tokens.css`), the AI is a 2D drawing on a canvas (`orb/Orb.tsx`), the map is a grid of squares (`components/RegionMap.tsx`), the logo is five shapes and sixteen symbols (`components/LogoBuilder.tsx`), and challenges stream as fast as the AI answers (`games/useMinigame.ts`, `windows/BattleWindows.tsx`).

Success, in the owner's terms: opening the game feels like walking into your own AI lab; every screen is a bench of instruments you operate, not a page you scroll; the things that matter (your AI, the country, your emblem, your trophies, the challenge boards) are real 3D objects; and a challenge is something you sit back and watch.

## 2. Goals, non-goals, anti-goals

**Goals**
- G1 The B look on every screen, window, toast and the world chat, on both dark and light glass.
- G2 The pixelated glass with the aurora behind every screen, in the company's colors (C57).
- G3 No page scrolling anywhere in a world: each screen fits the window as a bench.
- G4 Real 3D: the AI core, the emblem badge, the US map with its markers, the trophies, the challenge boards and dice, the data center and the graphics cards.
- G5 The lab door, the founding bench and the emblem maker (C58, C61, C62).
- G6 Challenges paced to about a minute, with faster and skip (C59).
- G7 A final pass that finds and redoes generic design (B7), with evidence.

**Non-goals**
- No change to what any control does, to game rules, balance, numbers, training or the AI's answers. Every behaviour and every control stays (R1, R4, R5).
- No sound or music (C47).
- No phone layout (the game targets desktop and laptop PCs; it must still work, stacked, on a small window).
- No change to how worlds, friends, helpers or Fair mode work.

**Anti-goals (a failure even if it works)**
- The look reads as a template: identical rounded cards, glassmorphism for its own sake, purple-blue gradients, tracked-out capital eyebrows over every heading, emoji as icons, a monospace face for every small label. (The engraved plate labels and keycap lettering of the chosen look, A1, are hardware engraving the owner picked, not eyebrows: one engraved name per module and on the controls themselves. Capitals anywhere else, such as inside displays, in body text or as decoration above text, are generic and get removed.)
- The 3D or the glass slows the player's own AI training or makes the game stutter on an RTX 3050 class card.
- Text becomes harder to read than today on either glass.
- A challenge is "slow" by padding: dead waiting with nothing to look at.
- Anything that leaks another player's private data (R2) through a new visual.

## 3. Constraints

- **R1:** the AI's words are shown as written. Pacing may delay showing them; it never edits, cuts or reorders them.
- **R2:** visuals of another company use only what its public view already shows (palette, logo, public scores, public counts).
- **R3, shared card:** each player's graphics card trains their AI while they play. The 3D and the glass must leave it free: draw only what is on screen, rest when nothing moves, cap the frame rate, stop when the tab is hidden, and offer a "lighter graphics" switch (§11).
- **R5:** the AI core's look stays a measurement (same inputs as `orb/params.ts`, smooth functions, no tiers or unlocks).
- **R7:** plain words everywhere; the hardware look must not bring in technical labels.
- **Accessibility and tests:** every control keeps its role and accessible name (the end-to-end tests in `tests/e2e/` and the unit tests find controls by role and name). Where a name must change, the tests change with it. Keyboard focus stays visible; reduced motion stays honored (`motion.ts`).
- **No network at play time:** fonts and data are bundled with the game (friends play through a tunnel; nothing may be fetched from a font or map service).
- **Licenses:** fonts under the SIL Open Font License, map boundaries from a public-domain source, the 3D engine under an open license; each recorded with attribution in the repository.
- **Old worlds:** worlds made before the redesign keep working; their companies' old logos and palettes display correctly.
- **Browsers:** current Edge and Chrome on Windows (the e2e tests use Edge). Without 3D support the game still works with flat stand-ins (§8.0).

## 4. Approaches

1. **Benches with a 3D layer (chosen).** Every screen is a full-window bench of hardware modules over the glass; long lists scroll inside their own display; switching screens turns the station knob and slides the next bench in; windows are metal panels that slide out; one 3D engine draws the objects where they matter. Keeps text sharp, keeps the card free, keeps every control a real control.
2. **A walk-through 3D lab.** The whole game as one 3D room with a moving camera. Most immersive, but reading and typing (chat, sliders, tables, the AI's replies) get blurry and slow in 3D, it costs the most graphics card time, and it would take several times longer. Dropped.
3. **A new coat of paint.** Today's layouts with new materials and 3D objects dropped in. Fastest, but the screens would still scroll like web pages, which is what B3 asks to remove. Dropped.

## 5. The look: "Instrument bench"

### 5.1 Materials

- **Bench metal.** Modules are machined plates with four screws in the corners, a fine brushed texture, a light top edge and a dark bottom edge, and a soft drop shadow onto the glass. Dark glass gets dark anodized metal; light glass gets light aluminium.
- **Displays.** Readouts sit in recessed dark displays (always dark, on both glasses) with an inner shadow. Numbers and short readouts glow in the company's lamp color in a dot-matrix face; longer text inside displays is light text in the bench face.
- **Engraved labels** name each module and control (A1).
- **Lamps.** Small round lamps show state: lit in the lamp color for "on", green for ready or done, amber for waiting, red for a problem. Warnings that need the player (the rescue, an overloaded service) blink slowly.
- **Hierarchy through hardware.** Modules differ in size, depth and finish by importance (the cradle of the AI core is the deepest and heaviest; small status modules are thin plates). Not every module is the same rounded card.

### 5.2 Type

- **Barlow Semi Condensed** (OFL) for all text: engraved labels, body text, buttons.
- **Doto** (OFL, dot-matrix) only inside displays, for numbers and short readouts (money, counts, dates, scores, names on name plates).
- Both bundled with the game. A clear type scale with fixed-width numbers in displays, so values never shift the layout as they change.

### 5.3 Color roles

- **Main color:** the glass tint, the lamp color, the active controls, the AI core.
- **Second color:** the aurora, rings and secondary data.
- **Glass (C57):** light glass when the main color reads better on white than on black (`contrast(main, white) >= contrast(main, black)`), dark glass otherwise.
- **Lamp color:** the main color on dark glass; the main color lifted toward white on light glass (so it glows inside dark displays).
- **Text on the glass** (outside displays) uses the ink of that glass (near-black on light, near-white on dark); accent text uses the main color, which always reads on its own glass by C57.
- **Other companies' colors** (rival names, map regions, chart lines, logos in tables) are adjusted for the viewer's glass when they would not read on it: lightened on dark glass, deepened on light glass, keeping the hue (A3).
- **Status colors** (good, warning, danger) are fixed per glass and never replaced by a company color.

### 5.4 Controls (each keeps the role and name of the control it replaces)

| Today | On the bench |
|---|---|
| Primary button | A push button with a lit ring in the lamp color; it travels down when pressed. |
| Secondary button | A plain push button. |
| Toggle and checkbox | A toggle switch (lever up or down) with its lamp. |
| Radio and tab groups | A rotary selector (positions engraved around a knob) or a row of illuminated keys, by count: up to five options as keys, more as a rotary selector. |
| Slider and number input | A slider with a machined cap in a recessed track, or a knob (drag up or down, mouse wheel, arrow keys), with its value in a small display. |
| Text field | A display you type into (dot-matrix for short names and codes, the bench face for longer text and chat). |
| Select (dropdown) | A rotary selector, or a display with step keys when the list is long (rival lists, symbol lists). |
| Link-like actions | Small push buttons, never underlined text. |

Every hardware control works with the keyboard (focus ring in the lamp color, arrows step knobs and sliders, Enter or Space press buttons and flip switches) and keeps a native element underneath for screen readers and tests.

### 5.5 Motion

Mechanical and purposeful: knobs move in detents with a slight overshoot, push buttons travel, switches snap, panels slide on rails, displays power on line by line, lamps fade up. One orchestrated moment per screen arrival (the bench slides in, then its displays power on), never a fade-and-rise on every element. Motion answers the player's actions; ambient motion is limited to the aurora, the AI core and lamps. Reduced motion: everything still works, with movement replaced by instant changes and a still glass.

## 6. The glass and the aurora (B4, C57)

- A full-window layer behind everything: the main color's glass, made of small square glass tiles (about 15 pixels at 1440 wide) with a bevel, thin grout lines, a slight per-tile variation, a slow band of light passing across, and a soft lamp that follows the pointer behind the glass.
- Behind the glass, nothing but light: two or three slow aurora curtains of the second color with soft rays, seen through the tiles (each tile shows the light behind its center, so the aurora is "pixelated").
- Dark glass: deep tinted near-black glass, the aurora glowing through. Light glass: frosted near-white glass tinted with the main color, the aurora as a soft colored wash, never dark enough to hurt text.
- Before a company exists (the lab door), the glass uses the neutral palette (`theme/palette.ts`); during founding it follows the color dials live, switching between light and dark as the main color crosses the rule; founding the company "powers up" the glass (the aurora brightens, then settles).
- Cost: drawn at reduced resolution (the tiles hide it), at most 30 frames a second, paused when the tab is hidden; a single still frame under reduced motion or "lighter graphics"; a static CSS rendering of the same colors when 3D is unavailable.

## 7. Layout: benches, not pages (B3, G3)

### 7.1 The station frame (inside a world)

- **Top strip** (one long plate): the company emblem and name, the world name, displays for cash, profit this month and the date, the speed keys (pause, 1×, 2×, 4×), the day progress bar, and the "your computer" status with its lamp. Everything the current `TopBar` shows stays, except the overload warning, which moves to a hazard plate under the strip (§7.3).
- **Station knob** (left column): a large knob above an engraved list of the nine stations (Company, Market, Money, AI Lab, Playground, Benchmarks, Rivals, Arena, AI Directory), each with a lamp; picking a station turns the knob to it. Under it, the "Lighter graphics" switch and the world chat intercom.
- **The bench:** the rest of the window, filled by the station's modules.
- **Changing station:** the knob turns, the old bench slides out sideways, the new one slides in and its displays power on. The glass stays still behind it.
- **The page never scrolls.** Long content (the feed, chat, tables, version lists, report cards) scrolls inside its own display, with a display-style scroll bar.
- **Sizes:** designed at 1440×900. The bench scales between about 1280 and 1920 pixels wide. Below about 1280 wide or 720 high, modules stack into fewer columns and the bench area scrolls inside the frame (the strip and knob stay put).

### 7.2 Windows become slide-out panels

All 30 window kinds (`store/windows.ts`) keep their content, their stacking and their Back and Close behaviour, and still grow out of what was clicked (C30). They become heavy metal panels that slide out of the bench over a dimmed glass, with an engraved title plate, push buttons for Back and Close, and their content in modules and displays.

### 7.3 Other overlays

- **Toasts** print out of a slot under the top strip as paper-like slips with a lamp color by kind, and slide away.
- **Warnings that need the player** (the rescue offer, servers leaving questions unanswered) are hazard plates under the top strip, each with a blinking red lamp and its push button.
- **World chat** is an intercom module that slides up from the station column: a speaker grille, a display with the messages, a typing display and a "send" push button.
- **The Victory ceremony** (`surprises/Goals.tsx`) keeps its content in a large panel with the emblem badge and trophies in 3D.

## 8. The 3D objects (B5, G4)

### 8.0 One 3D engine

One shared 3D engine (three.js) draws every object; each object appears in its own place in the layout and is drawn only while visible and changing. Objects share materials (machined metal, enamel, glass, emissive light), one lighting setup and one glow pass. Every object has a flat stand-in for computers without 3D support and for unit tests (which run without a screen), so nothing depends on 3D to work. Objects are built in code (procedural models), except the map, which is built from public-domain state boundaries.

### 8.1 The AI core

- A faceted crystal with a glowing heart inside (light gathered through the crystal), held by geared metal rings that turn: the look picked in C60.
- Every property is a measurement, mapped from today's `orbParams` inputs (R5): size and brightness from the overall score, one ring per skill that has crossed its threshold plus the read-words ring, ring detail from words read, the rings' spin from boldness and inventiveness, wobble from inventiveness, breathing calmness from steadiness, sparks from verbosity, glow softness from warmth, flicker from the slip rate. Before the first AI, a small spark in an empty cradle.
- While training: streams of light in each dataset's color flow into the crystal in proportion to the mix; brightness rises as the loss falls (today's `training` input).
- New customers today drift in as sparks in their groups' colors (today's `arrivals`).
- The six bought orb looks (`LOOK_NAMES` in `spending/Spending.tsx`: Classic, Nebula, Crystal, Plasma, Aurora, Supernova) become 3D treatments around the core (a drifting color cloud, a faceted outer ring, a hot breathing glow, northern-light ribbons, bursting rays); they never change the measured core.
- Drag to turn it; it settles back to a slow turn. Appears on the Company bench (large), in the AI and product windows, in the AI Lab's training chamber, and in the Versions rack (small, one per version).

### 8.2 The emblem badge (C58)

- A 3D badge: the frame extruded with depth, the face in the chosen finish, the rings and pattern on the face, the symbol or letters raised, a light sheen that moves as it turns.
- Shown in 3D at about 64 pixels and up (founding press, top strip, rival plates, the arena podium, news slips' large cards); smaller uses are flat renders of the same emblem (A4).
- The four bought logo motions (`logo:still`, `pulse`, `shimmer`, `orbit`) become motions of the badge: still, a heartbeat pulse, a light sweep, a spark circling the frame.

### 8.3 The US map

- The real shapes of the 50 states and DC, grouped into the game's nine regions (`regions.ts`), as a 3D tabletop model with Alaska and Hawaii as insets. Source: public-domain US Census boundaries (bundled, simplified).
- Each region's top glows in the color of the company with the most users there (same for everybody, B5). Height shows how much of the region's willing people use the leader's AI. A region nobody leads yet is plain glass.
- Markers: data centers as small lit buildings (with scaffolding while being built), ad campaigns as pulsing rings on the region, sponsored events as pins with the event's mark.
- Controls: drag to turn, drag with the right button or the mouse wheel to tilt and zoom within limits, a push button to reset the view. Hover lights the region and shows its readout (its name, the leader, your customers there and your share); clicking opens the region window, as today.
- Every region also stays reachable as a real button (keyboard and tests), as today.
- Appears on the Market bench (large), the Company bench (small, also turnable) and the rival and region windows.

### 8.4 Other objects

- **Trophies:** each earned milestone a small 3D trophy on a lit shelf, in metal and lamp light. Only earned trophies appear, as today: never outlines of what is left to earn (R5, `surprises/Trophies.tsx`).
- **The data center** (its window): a 3D building model that lights up floor by floor as it is built.
- **Graphics cards** (the computers and cards windows): each card as a 3D card with its fans turning while it works.
- **Challenge boards:** see §10.
- **Looks (C65):** the orb styles are worn by the 3D core itself (its rings, light and aura), tried on live in a showroom before buying; the logo motions move the 3D badge.

### 8.5 Pictures (C63)

- Every emoji in the client is replaced by a picture from one set in one style: bold silhouettes that read when engraved into metal (dice faces, trophies), lit as dots or light on displays (puzzles, marks), and as flat marks at small sizes. That covers the Story Dice heroes, situations and words (`games/pictures.tsx`), the pattern puzzles' animals, the trophy marks (`surprises/Trophies.tsx`), the sponsored-event marks (the map pins), the game marks and any other emoji left (toasts, buttons).
- The words that go with each picture stay as they are (R1, R7): a picture never replaces the words the AI chose or the plain name of a thing.
- Source: an open-licensed game picture set, with every picture's author and license recorded in the repository (ADR 0011 §8).

## 9. The lab door and founding (B2, C58, C61, C62)

### 9.1 Before a world

- **Sign in (`Welcome`):** an ID badge printer: type your name into its display, press "Continue" and a badge with your name prints and slides out; the door opens to your worlds.
- **Your worlds (`Home`):** a wall rack of lit world panels (name, day or "not started", players), each with its "Open" push button; worlds made with an older version show as dim panels with their note.
- **A new world:** a small bench with a name plate display, the goal as a rotary selector (the options of `GoalPicker`), the Fair mode toggle switch with its notice on a plate, and a "Start" push button.
- **Join a friend:** a keypad with a code display and a "Join" button; the join link (`/join/<code>`) lands on the same keypad with the code filled in.

### 9.2 Founding (`Lobby`, C61)

Exactly as in `founding.html`:
- Name plate (dot-matrix display), two hue dials with brightness sliders and twelve quick-color cartridges (the twelve `PALETTE_PRESETS`), and a line that says which glass the colors give.
- The emblem press: the live 3D badge in a chamber, drag to turn.
- The deck: knobs for Frame, Corners, Turn, Rings, Glow; symbol keys (the sixteen symbols and Letters); sliders for ring weight and symbol size; switches for Pattern and Finish.
- Players here (lamps for who is online), Invite friends (the code in a display, "Copy link").
- The lever: pull it down (or press it with the keyboard) once the name has enough characters; the badge is stamped, the glass powers up, and the bench shows "founded, waiting for the host" or, for the host, the big "Open for business" switch.

### 9.3 The emblem (C58)

An emblem is: frame (circle, corners, shield, diamond, rosette), corners (3 to 12), turn (0 to 355 degrees), roundness, rings (0 to 3), ring weight, glow, symbol (the sixteen existing symbols, or letters: up to three capitals or digits, by default the company's initials), symbol size, pattern (none, rays, grid, dots, waves) and finish (metal, enamel, glass, light). Old logos (`shape`, `symbol`, `initials`) map to emblems that look like them (a rounded square becomes four corners turned 45 degrees with roundness). The server checks every field, as it checks logos today.

### 9.4 Colors

The founding bench offers any color (C57 makes every main color readable on its own glass). The second color has no readability rule; where it is used as text it follows §5.3.

## 10. Every station as a bench (G1, G3)

Each station keeps all its content and controls; this section says how they are arranged and what replaces what. Module names are the engraved labels.

- **Company:** the AI core in its cradle at the center, with its name and score display, "Train", "Chat" and "Launch" push buttons and "Customize your look"; Customers module (count display, this month's change, a group meter and its group keys that open each group); Money module (cash and profit displays, a trace display of profit by month, the latest report card as a stamped card); the feed as a ticker display; the small 3D map; the lineup of products as plates; the arch-rival plate; the drawer bay with Team, Licenses, Partner apps, Goal and Trophies (A2).
- **Market:** the large 3D map at the center; displays for people willing to use an AI, your customers and your share; the crowd as a dot-matrix display where each dot is a person; the growth trace; the ad campaigns module with its push button.
- **Money:** Coming in and Going out as two ledgers in displays, each category a key that opens its window; loans and investors; the investment fund with a trace of the portfolio growing or shrinking (C66); report cards as a rack of stamped cards; computers; profit by month as a bar display; cash day by day as a trace; your AIs on sale as plates; plans and prices with switches and sliders.
- **AI Lab:** the five steps (Data, Prepare, Design, Train, Versions) on a five-position rotary selector; the summary bar (cost, game days, real minutes, "Start training") as a long plate at the bottom.
  - Data: a mixing desk with one vertical fader per dataset and a level meter, the share in a display above each, "See examples" keys opening the example display.
  - Prepare: the cleaning options as toggle switches with lamps, and the before-and-after examples in displays.
  - Design: Grow from, Brain size, Attention span, Training length and Learning style as rotary selectors and knobs, with their plain-language explanations on plates.
  - Train: the training chamber with the live 3D core and its data streams, the loss trace on an oscilloscope display, "What it's learning" and "What it says" displays, a progress gauge, the run status lamp.
  - Versions: a rack of version cartridges (label, small core, scores), each opening its AI window.
- **Playground:** the chat as a terminal display (your messages and the AI's real replies, coaching thumbs as small keys under each reply), the games as a challenge selector leading into the challenge chamber (§10.1), and the interview mode.
- **Benchmarks (C64):** one global graph of every company's AIs over time, with a key per company to hide or show its AIs and a selector for the overall score or one category; your chosen AI's own scores read beside it (the overall score display and its change since the AI it grew from).
- **Rivals:** one plate per rival (emblem, name, AI, public scores, customers), "Try" push buttons, each opening its window.
- **Arena:** the league tables as a scoreboard display with a rotary selector for the table; Battles as a list display with "Start a battle"; the battle itself in the challenge chamber.
- **AI Directory:** Shared by players, License offers, Ask a player and Rivals' AIs on sale as catalog displays with push buttons.

### 10.1 Challenges at a watchable pace (B6, C59)

- **A challenge chamber:** a large display bay where the game's board stands in 3D, with the score, lives and best in displays, and a control row: "Faster" (toggles double speed) and "Skip to the result", plus "Leave" or "Play another" as today.
- **Pacing:** the worker still plays at full speed; the client holds every step it receives and plays them back one at a time: the question appears, the AI "thinks" (its sureness gauges move to its real values), its answer types out at reading speed, the verdict lamp lights (with the right answer when it missed), then the board moves. About five seconds per step, so a ten-step game takes about a minute; Story Dice spends its minute on three stories typed out; a battle plays round by round, both answers side by side.
- **The result** shows only when the playback reaches the end (or on "Skip to the result"). Messages and news about the run (showcase press, ranking points) wait until the player has watched it, if they are watching.
- **Boards:** Puzzle Maze, a 3D tile path the AI core travels, doors at each junction; Pattern Bridge, glass tiles over a chasm that hold or crumble; Quiz Show, a stage with four answer lamps and a lit prize ladder; Story Dice, three real dice rolled for hero, situation and word, with an applause meter; Simon Says, command cards that flip up on a board and light green or red.
- **Duels:** both boards side by side, stepping together.
- Answers are only delayed, never changed (R1).

## 11. Performance and the quality floor

- **Frame budget:** the glass and all visible 3D objects together stay within a light load on an RTX 3050 laptop-class card while it trains: reduced-resolution glass, a 30-frame cap for ambient motion (60 while the player drags or a challenge plays), rendering only visible, changing objects, nothing while the tab is hidden.
- **Lighter graphics** (a switch at the foot of the station column, remembered by the browser when the player sets it): a still glass, no glow pass, simpler 3D, no ambient motion. The game turns it on by itself, with a note, when frames take too long.
- **Readability:** body text at least as readable as today on both glasses; displays meet strong contrast; focus rings visible.
- **Reduced motion:** honored everywhere (`motion.ts`), including the glass and the 3D.
- **Small windows:** §7.1.

## 12. The generic-design hunt (B7, G7)

The last phase, after everything is built and working:

1. Screenshot every station, window, toast and challenge on dark and light glass in the `check` game (`data-check/`, ports 8788/5175).
2. Review them with every design tool available: the Design plugin's design critique, UI Design Review (review, compare, polish), the audit suite's design-engineering, "make interfaces feel better" and animation lenses, the frontend-design checklist of generic tells (§2 anti-goals), and the 3D visual scorecard for the objects.
3. List every finding, most important first, fix them, screenshot again, and repeat until the reviews find nothing generic of importance.
4. Record the findings and fixes in the progress log.

## 13. Changes outside the client

- **Protocol:** the emblem (§9.3) as the logo type, accepting old logos; the palette rule of C57 (no readability refusal for new companies).
- **Sim:** the contrast helpers the glass rule and §5.3 use (shared with the client); the fund's balance in each day's sample for the portfolio trace (C66); no rule or balance change.
- **Server and protocol (C64):** a public record of every company's released or shared AIs (day made, scores), kept after retirement, for the global graph; an older world's day samples without the fund read as no history yet.
- **Server:** validation of the new emblem fields; old logos and rival packs' logos display through the mapping.
- **Docs:** this design, an ADR for the technical choices (3D engine, fonts, map data, pacing), the progress log, and a note in `docs/handoff/next-steps.md` that the redesign comes before the game document.

## 14. Done means

- The owner has seen and approved the new look in the running game (the `check` game).
- `npm test`, `npm run check` and `npm run e2e` pass.
- The work is committed and pushed on `m2-two-friends`.
- `docs/progress.md` has an entry, and `docs/handoff/next-steps.md` notes that the redesign comes before the game document.

## 15. Left to the builder

The exact sizes, spacings and type scale; the module arrangement inside each bench (within §10); the procedural models' details; the map's simplification level and projection; how the one 3D engine shares its drawing surface; the pacing's exact timings within "about a minute"; the automatic "lighter graphics" threshold; the order of work, provided the owner sees the first finished stations early and the generic-design hunt comes last.
