# ADR 0011: the look redesign ("Instrument bench"), technical choices

Date: 2026-11-26. Design: `docs/superpowers/specs/2026-11-26-look-redesign-design.md` (owner decisions C57–C62). The owner is not technical; these choices are Claude's, recorded here with their reasons.

## 1. The 3D engine: three.js, one shared renderer

- **Choice:** three.js (MIT), used directly from small React wrappers; no React-three-fiber or helper libraries.
- **Why:** the objects are few and specific (core, emblem, map, trophies, boards, data center, graphics cards); a thin wrapper gives full control of when to draw, which matters because the player's own graphics card is training their AI at the same time (R3). Fewer dependencies, fewer React version couplings.
- **One renderer:** browsers allow only a handful of live 3D drawing contexts, and windows can show several objects at once. One renderer draws each visible object into its own on-page canvas (draw, then copy), so layering with windows and toasts stays ordinary page layering.
- **When it draws:** only objects that are on screen and changing; ambient motion capped at 30 frames a second, 60 while dragging or during a challenge; nothing while the tab is hidden.
- **Without 3D** (no WebGL, unit tests under happy-dom): every object has a flat stand-in, so the game and its tests never depend on 3D.

## 2. The glass: its own small shader, not three.js

- **Choice:** the pixelated glass and aurora are one full-window fragment shader on a plain WebGL canvas (as in the mockups), drawn at reduced resolution and at most 30 frames a second.
- **Why:** it is a single full-screen effect with no geometry; three.js adds nothing. Reduced resolution is invisible behind the tiles. A static CSS rendering of the same colors stands in without WebGL; a still frame under reduced motion and "lighter graphics".

## 3. Fonts: bundled, open licensed

- **Barlow Semi Condensed** (`@fontsource/barlow-semi-condensed`, OFL-1.1) for all text; **Doto** (`@fontsource/doto`, OFL-1.1) for display readouts. Bundled by the client build; nothing is fetched from a font service at play time (friends connect through a tunnel).

## 4. The map: public-domain boundaries, projected in advance

- **Choice:** `us-atlas` (ISC; its `states-albers-10m.json` is derived from the US Census Bureau's cartographic boundary files, which are public domain) read with `topojson-client` (ISC). The states are already projected (Albers USA, with Alaska and Hawaii as insets), so no projection library is needed. Each state's outline becomes an extruded 3D shape, grouped into the game's nine regions (`apps/client/src/regions.ts`).
- **Why:** real shapes (B5), a public-domain source, no network at play time, small size.
- **Attribution:** recorded in the client's licenses note.

## 5. Procedural models

- **Choice:** the AI core, the gears, the emblem badge, trophies, the data center, graphics cards, maze, bridge, quiz stage, dice and Simon board are built in code from three.js geometry and shared materials.
- **Why:** no external generator keys exist on this PC (Tripo, Gemini), no binary model files in the repository, and every object can follow the company's colors and the measurements (R5) directly.

## 6. Challenge pacing in the client

- **Choice:** the worker keeps playing at full speed; a pure client-side "director" holds the streamed steps and plays them back at about five seconds a step, with double speed and skip. Tested as pure logic.
- **Why:** pacing is presentation (R1: answers are only delayed); no server, protocol or worker change; old runs and history are unaffected.

## 7. The emblem and the palette rule

- **Emblem:** the logo type gains the emblem fields (design §9.3); old logos stay valid and are mapped to look like themselves. The server validates every field.
- **Palette (C57):** the glass is chosen from the main color (light glass when the main color reads better on white than on black), so any main color reads on its owner's glass; the founding rule that refused hard-to-read colors is removed for new companies. Other companies' colors are adjusted for the viewer's glass where they would not read (design §5.3).

## 8. Pictures (C63): game-icons.net, extracted once

- **Choice:** the pictures come from game-icons.net (CC BY 3.0, by Lorc, Delapouite and other contributors), read from the `@iconify-json/game-icons` package at development time only. A small script copies the roughly 90 chosen pictures into a generated source file in the client, with each picture's name, author and license; the license note in the client credits them. The package is a development dependency and is not shipped.
- **Why:** bold, game-style silhouettes that engrave well on 3D dice and trophies and read on dot displays; one consistent style for all 90; an open license that only asks for credit. The common interface icon sets (Lucide, Phosphor, Tabler) were passed over: they read as generic app icons, which design §2 counts as a failure.
