# Immersion pass (C30) — design

**Date:** 2026-10-28 · **Owner decision:** C30 (game-layer design §1) · **Applies to:** the Phase 1 screens now; every Phase 2+ screen follows the same rules.

## Goal

After playing Phase 1 the owner asked for a game that feels engaging and alive: "I do not want the player to just simply look at graphs. I want there to be motion, transition, ability to open up windows, click on stuff, expanding graphs, visuals." Then straight into Phase 2 with no review stop.

## Principles

1. **Motion means something.** Things move because something happened: a day passed, customers arrived, money came in, a post appeared, a screen opened. The orb's idle life is the only constant motion.
2. **Every number has a story.** Any number, bar, face, card, orb or post on a screen can be clicked, and opens a window that explains it with its history.
3. **Real data only.** Every visual is drawn from the view. Nothing is invented (R1, R5); private history stays in its owner's view (R2).
4. **Never in the way.** Windows don't pause the shared clock and update live while open. Escape, the backdrop or the close button always gets out.
5. **Reduced motion respected.** When the system asks for less motion, every animation is instant (the orb draws one still frame, as before).

## Building blocks

- **Motion library.** `motion` (already a dependency): springs, enter and exit, shared layout. `MotionGlobalConfig.skipAnimations` makes every animation instant under reduced motion and in tests.
- **Windows.** One host in the app shell shows a stack of windows. A window grows out of the thing that was clicked and shrinks back on close. Clicking inside a window opens the next one on top, with a back arrow. Escape goes back one; the backdrop closes all; changing screens closes all; focus returns to what opened it. Kinds: customers, money, customer group, product, AI, month, money category, post.
- **Big charts.** Own SVG charts for the game's numbers (uPlot stays for the live training charts): a day chart (lines or stacked areas, ranges 30 days / 90 days / all, a hover readout of every series for that day, drawn in on open) and month bars (above and below zero, each bar clickable, a hover readout).
- **Day history.** The sim records one sample per company per day (cash, customers per group, paying customers, followers, reputation) and keeps 120 days. The owner's view carries it keyed by day, so each new day patches as one entry (the patch diff resends a whole array whenever its length changes). Worlds saved before history existed keep working; their history starts on their next day.
- **Feel.** Screens slide in with their panels staggered; anything clickable lifts and glows on hover; the screen list has a sliding highlight; a thin bar under the top bar fills over each game day (still while paused); numbers that change float a small "+12" or "−$300"; new posts and news slide into the feed; news about the player's own company pops up as a toast, and milestones (launch, first paying customer, customer marks, going viral) burst into a small celebration.

## Screens

- **Shell.** Day bar, floating cash change, cash opens the money window, toasts and celebrations, sliding screen highlight, screen transitions.
- **Company.** The orb opens the AI window and new customers stream into it as colored sparks, one color per group. Customers open the customers window; each group in the bar opens its group window. Cash and profit open the money window. The profit line expands. Lineup products open the product window. Feed items slide in; a post opens the post window.
- **Market.** A live crowd: your customers as dots clustered by group, each dot a fixed number of people that steps up (1, 10, 100, …) as you grow; new dots pop in. A cluster or a group card opens the group window: its customers over time, plans, mood, its spotlight person and its latest posts.
- **Money.** Bars grow in. Every income and cost row opens that category month by month. Profit bars are clickable months (the month window: in, out and customers at the close). The profit panel expands into the money window (cash day by day plus monthly profit).
- **AI Lab (training).** The two live charts expand in place to full width and double height.

## Decisions (technical; recorded here, no owner question needed)

- **I1 `motion` for presence and layout.** Exit animations and shared layout are hard to get right by hand. It is already installed and works with React 19. Cost if wrong: about 40 kB more script.
- **I2 History keyed by day in the view.** The diff sends an object's new key as one patch; an array that grows or shifts would be resent whole every day. Cost if wrong: one conversion on the client.
- **I3 120 days of history.** That is four game months day by day; months beyond come from the ledger. Cost if wrong: a constant.
- **I4 One window stack per world screen.** Changing screens closes it; windows are not in the address. Cost if wrong: no deep links to a window.
- **I5 Own SVG charts.** They are small, testable in the DOM, animatable and themed by the company's colors. Cost if wrong: some code that a chart library could have provided.
- **I6 Old sim-2 worlds keep working.** The history field is filled from the next day on. No sim version bump for this pass; Phase 2 bumps to sim-3 anyway.

## Testing

- **Sim.** One sample per day, the 120-day cap, whole numbers, and history only in the owner's view (R2 property).
- **Client.**
  - Windows open from clicks and stack, and close with Escape, back and the backdrop.
  - Charts show the right points per range and the hover readout.
  - Each screen's clickable things open the right window with the right numbers.
  - The crowd draws one dot per step of people.
  - Toasts appear only for new news about the player's own company.
  - The orb's arrivals come from new customers.
- **Browser.** A real game: open every window, watch customers arrive, and check reduced motion.
