import {
  type Emblem,
  type EmblemFrame,
  LOGO_SYMBOLS,
  type LogoShape,
  type LogoSymbol,
  type Palette,
  TEXT_LIMITS,
  type WorldView,
} from "@airace/protocol";
import { PALETTE_PRESETS } from "@airace/sim";
import { useEffect, useState } from "react";
import { Cartridge } from "../bench/Cartridge.tsx";
import { Display } from "../bench/Display.tsx";
import { HueDial } from "../bench/HueDial.tsx";
import { KeyRow } from "../bench/KeyRow.tsx";
import { Knob } from "../bench/Knob.tsx";
import { Lamp } from "../bench/Lamp.tsx";
import { Lever } from "../bench/Lever.tsx";
import { Module } from "../bench/Module.tsx";
import { PushButton } from "../bench/PushButton.tsx";
import { Slider } from "../bench/Slider.tsx";
import { symbolArt } from "../emblem/art.ts";
import { EmblemBadge } from "../emblem/EmblemBadge.tsx";
import { useSocket } from "../store/session.ts";
import { applyBenchTheme, glassFor, powerUpGlass } from "../theme/glass.ts";

const FIRST_PRESET = PALETTE_PRESETS[0]?.palette ?? { primary: "#4f8cff", secondary: "#38d9c0" };

const FRAMES: EmblemFrame[] = ["circle", "corners", "shield", "diamond", "rosette"];
const FRAME_NAMES: Record<EmblemFrame, string> = {
  circle: "Circle",
  corners: "Corners",
  shield: "Shield",
  diamond: "Diamond",
  rosette: "Rosette",
};

/** The emblem a new company starts from (as the approved mockup, C61). */
const START: Omit<Emblem, "letters"> = {
  frame: "corners",
  corners: 6,
  turn: 0,
  roundness: 0,
  rings: 2,
  ringWeight: 3,
  glow: 40,
  symbol: "spark",
  symbolSize: 100,
  pattern: "rays",
  finish: "metal",
};

/** The first letter or digit of up to three words of the name, in capitals; "A" when none. */
export function lettersOf(name: string): string {
  const letters = name
    .split(/\s+/)
    .map((w) => w.toUpperCase().replace(/[^A-Z0-9]/g, "")[0] ?? "")
    .join("")
    .slice(0, 3);
  return letters || "A";
}

/** The first logos' fields closest to an emblem, for older clients that only read those. */
export function logoFieldsOf(e: Emblem): {
  shape: LogoShape;
  symbol: LogoSymbol | null;
  initials: string;
} {
  const shape: LogoShape =
    e.frame === "corners"
      ? e.corners === 4
        ? "rounded-square"
        : e.corners === 6
          ? "hexagon"
          : "circle"
      : e.frame === "rosette"
        ? "circle"
        : e.frame;
  return { shape, symbol: e.symbol, initials: e.letters.slice(0, 2) };
}

// Colors as hue and lightness for the dials (saturation kept from the color picked).
type Hsl = { h: number; s: number; l: number };

function toHsl(hex: string): Hsl {
  const n = Number.parseInt(hex.slice(1), 16);
  const r = ((n >> 16) & 255) / 255;
  const g = ((n >> 8) & 255) / 255;
  const b = (n & 255) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0;
  let s = 0;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    h = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
    h *= 60;
  }
  return { h: Math.round(h) % 360, s: Math.round(s * 100), l: Math.round(l * 100) };
}

function toHex({ h, s, l }: Hsl): string {
  const sat = s / 100;
  const lig = l / 100;
  const k = (n: number) => (n + h / 30) % 12;
  const a = sat * Math.min(lig, 1 - lig);
  const f = (n: number) => lig - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return `#${[f(0), f(8), f(4)]
    .map((x) =>
      Math.round(x * 255)
        .toString(16)
        .padStart(2, "0"),
    )
    .join("")}`;
}

/** Founding a company (C61, C5: its palette becomes the player's whole lab), then opening the world. */
export function Lobby({ view }: { view: WorldView }) {
  return view.me.company ? <OpenForBusiness view={view} /> : <FoundCompany view={view} />;
}

/** Everyone in the world, with who is here now. */
function Players({ view }: { view: WorldView }) {
  return (
    <Module label="Players here">
      <ul aria-label="Players" className="space-y-2">
        {view.members.map((m) => (
          <li key={m.userId} className="flex items-center gap-2.5 font-semibold">
            <Lamp state={m.online ? "ready" : "off"} label={m.online ? "here now" : "away"} />
            <span className="player-text">{m.displayName}</span>
            <span className="ml-auto text-sm font-medium text-muted">
              {m.online ? "online" : "away"}
            </span>
          </li>
        ))}
      </ul>
    </Module>
  );
}

/** The world's invite code and link (Stage 2). */
function InviteFriends({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  const link = `${location.origin}/join/${code}`;
  async function copy() {
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }
  return (
    <Module
      label="Invite friends"
      hint="Send them the link, or the code to type at their lab door."
    >
      <Display className="text-center">
        <p data-testid="invite-code" className="dotface dotface--lg m-0 tracking-[0.14em]">
          {code}
        </p>
      </Display>
      <p className="mt-2 break-all text-xs text-muted">{link}</p>
      <div className="mt-3 flex items-center gap-3">
        <PushButton variant="secondary" onClick={copy}>
          Copy link
        </PushButton>
        {copied && <span className="text-sm text-muted">Copied</span>}
      </div>
    </Module>
  );
}

function FoundCompany({ view }: { view: WorldView }) {
  const socket = useSocket();
  const [name, setName] = useState("");
  const [palette, setPalette] = useState<Palette>(FIRST_PRESET);
  const [main, setMain] = useState<Hsl>(() => toHsl(FIRST_PRESET.primary));
  const [second, setSecond] = useState<Hsl>(() => toHsl(FIRST_PRESET.secondary));
  const [emblem, setEmblem] = useState<Omit<Emblem, "letters">>(START);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [min, max] = TEXT_LIMITS.companyName;
  const ready = name.trim().length >= min;
  const full: Emblem = { ...emblem, letters: lettersOf(name) };
  const glass = glassFor(palette.primary);

  // The whole lab follows the dials while the company is being founded (C57).
  useEffect(() => {
    applyBenchTheme(palette);
  }, [palette]);

  const set = <K extends keyof typeof emblem>(key: K, value: (typeof emblem)[K]) =>
    setEmblem((e) => ({ ...e, [key]: value }));
  const dial = (which: "main" | "second", next: Hsl) => {
    if (which === "main") setMain(next);
    else setSecond(next);
    setPalette((p) => ({ ...p, [which === "main" ? "primary" : "secondary"]: toHex(next) }));
  };

  async function found() {
    if (!ready || busy) return;
    setBusy(true);
    setError(null);
    const logo = { ...logoFieldsOf(full), emblem: full };
    const r = await socket.command(view.world.id, { type: "createCompany", name, logo, palette });
    setBusy(false);
    if (!r.ok) setError(r.reason);
    else powerUpGlass();
  }

  return (
    <main className="founding-frame">
      <header className="founding-strip module module--thin">
        <div>
          <h1 className="m-0 font-display text-[25px] font-bold">Found your company</h1>
          <p className="m-0 text-sm text-muted">
            Its colors become your whole lab. You can't change them later.
          </p>
        </div>
        <div className="ml-auto text-right">
          <span className="engrave">World</span>
          <p className="player-text m-0 text-[17px] font-bold">{view.world.name}</p>
        </div>
      </header>

      <div className="founding-left">
        <Module label="Name plate">
          <input
            className="display-input founding-plate"
            value={name}
            maxLength={max}
            placeholder="Acme Intelligence"
            aria-label="Company name"
            autoComplete="off"
            spellCheck={false}
            onChange={(e) => setName(e.target.value)}
          />
          <p className="mt-2 text-[13px] text-muted">{`${min} to ${max} characters. Everyone in the world will see it.`}</p>
        </Module>
        <Module label="Colors" className="flex-1">
          <div className="grid grid-cols-2 gap-3">
            {(["main", "second"] as const).map((which) => {
              const c = which === "main" ? main : second;
              const label = which === "main" ? "Main color" : "Second color";
              return (
                <div key={which} className="flex flex-col items-center gap-2.5">
                  <HueDial
                    label={label}
                    hue={c.h}
                    color={which === "main" ? palette.primary : palette.secondary}
                    onChange={(h) => dial(which, { ...c, h, s: Math.max(55, c.s) })}
                  />
                  <Slider
                    label={`${label} brightness`}
                    value={c.l}
                    min={18}
                    max={82}
                    showLabel={false}
                    onChange={(l) => dial(which, { ...c, l, s: Math.max(55, c.s) })}
                  />
                </div>
              );
            })}
          </div>
          <p className="engrave mt-4">Quick colors</p>
          <div className="mt-2 grid grid-cols-6 gap-2">
            {PALETTE_PRESETS.map((p) => (
              <Cartridge
                key={p.id}
                label={p.label}
                primary={p.palette.primary}
                secondary={p.palette.secondary}
                pressed={
                  p.palette.primary === palette.primary && p.palette.secondary === palette.secondary
                }
                onClick={() => {
                  setPalette(p.palette);
                  setMain(toHsl(p.palette.primary));
                  setSecond(toHsl(p.palette.secondary));
                }}
              />
            ))}
          </div>
          <p className="mt-4 flex items-center gap-2.5 text-[13.5px] text-muted">
            <Lamp state="on" label="Lab glass" />
            {`Your lab glass is ${glass}: it follows your main color.`}
          </p>
        </Module>
      </div>

      <div className="founding-mid">
        <Module weight="heavy" className="founding-press">
          <div className="founding-chamber display" data-testid="preview">
            <span className="engrave founding-chamber__label">Emblem press</span>
            <span className="founding-chamber__hint">Drag to turn it</span>
            <EmblemBadge
              logo={{ ...logoFieldsOf(full), emblem: full }}
              palette={palette}
              size={260}
              label="Your logo"
            />
          </div>
        </Module>
        <Module className="founding-deck">
          <div className="founding-knobs">
            <Knob
              label="Frame"
              value={FRAMES.indexOf(emblem.frame)}
              min={0}
              max={FRAMES.length - 1}
              turn={240}
              format={(i) => FRAME_NAMES[FRAMES[i] ?? "circle"]}
              onChange={(i) => set("frame", FRAMES[i] ?? "circle")}
            />
            <Knob
              label="Corners"
              value={emblem.corners}
              min={3}
              max={12}
              onChange={(v) => set("corners", v)}
            />
            <Knob
              label="Round"
              value={emblem.roundness}
              min={0}
              max={100}
              step={5}
              format={(v) => `${v}%`}
              onChange={(v) => set("roundness", v)}
            />
            <Knob
              label="Turn"
              value={emblem.turn}
              min={0}
              max={355}
              step={5}
              turn={355}
              format={(v) => `${v}°`}
              onChange={(v) => set("turn", v)}
            />
            <Knob
              label="Rings"
              value={emblem.rings}
              min={0}
              max={3}
              turn={200}
              onChange={(v) => set("rings", v)}
            />
            <Knob
              label="Glow"
              value={emblem.glow}
              min={0}
              max={100}
              step={5}
              format={(v) => `${v}%`}
              onChange={(v) => set("glow", v)}
            />
          </div>
          <div className="founding-cell founding-cell--4">
            <span className="engrave" aria-hidden="true">
              Symbol
            </span>
            <fieldset className="founding-symbol-keys">
              <legend className="sr-only">Symbol</legend>
              {LOGO_SYMBOLS.map((s) => (
                <button
                  key={s}
                  type="button"
                  aria-label={s}
                  aria-pressed={emblem.symbol === s}
                  title={s[0]?.toUpperCase() + s.slice(1)}
                  className="symbol-key"
                  onClick={() => set("symbol", s)}
                >
                  <svg viewBox="0 0 100 100" aria-hidden="true" className="symbol-key__art">
                    <g
                      // biome-ignore lint/security/noDangerouslySetInnerHtml: fixed symbol art from emblem/art.ts
                      dangerouslySetInnerHTML={{ __html: symbolArt(s) }}
                    />
                  </svg>
                </button>
              ))}
              <button
                type="button"
                aria-label="Letters"
                aria-pressed={emblem.symbol === null}
                title="Letters"
                className="symbol-key symbol-key--letters"
                onClick={() => set("symbol", null)}
              >
                <span aria-hidden="true">Ab</span>
              </button>
            </fieldset>
          </div>
          <div className="founding-cell founding-cell--2">
            <Slider
              label="Ring weight"
              value={emblem.ringWeight}
              min={1}
              max={6}
              onChange={(v) => set("ringWeight", v)}
            />
            <Slider
              label="Symbol size"
              value={emblem.symbolSize}
              min={60}
              max={130}
              onChange={(v) => set("symbolSize", v)}
            />
          </div>
          <div className="founding-switches">
            <div className="founding-cell">
              <span className="engrave" aria-hidden="true">
                Pattern
              </span>
              <KeyRow
                legend="Pattern"
                options={[
                  { label: "None", value: "none" },
                  { label: "Rays", value: "rays" },
                  { label: "Grid", value: "grid" },
                  { label: "Dots", value: "dots" },
                  { label: "Waves", value: "waves" },
                ]}
                value={emblem.pattern}
                onChange={(v) => set("pattern", v)}
              />
            </div>
            <div className="founding-cell">
              <span className="engrave" aria-hidden="true">
                Finish
              </span>
              <KeyRow
                legend="Finish"
                options={[
                  { label: "Metal", value: "metal" },
                  { label: "Enamel", value: "enamel" },
                  { label: "Glass", value: "glass" },
                  { label: "Light", value: "light" },
                ]}
                value={emblem.finish}
                onChange={(v) => set("finish", v)}
              />
            </div>
          </div>
        </Module>
      </div>

      <div className="founding-right">
        <Players view={view} />
        <InviteFriends code={view.world.inviteCode} />
        <Module label="Found the company" className="flex flex-1 flex-col items-center">
          <Lever label="Found company" disabled={!ready} down={busy} onPull={() => void found()}>
            <p className="flex items-center gap-2 text-sm font-semibold">
              <Lamp
                state={busy ? "wait" : ready ? "ready" : "problem"}
                label={busy ? "Founding" : ready ? "Ready" : "Not ready"}
              />
              {busy
                ? "Founding…"
                : ready
                  ? "Ready: pull the lever down"
                  : "Name your company first"}
            </p>
          </Lever>
          {error && (
            <p role="alert" className="mt-3 text-center text-sm text-danger">
              {error}
            </p>
          )}
        </Module>
      </div>
    </main>
  );
}

function OpenForBusiness({ view }: { view: WorldView }) {
  const socket = useSocket();
  const [error, setError] = useState<string | null>(null);
  const company = view.companies.find((c) => c.id === view.me.company?.id);
  const isHost = view.world.effectiveHostUserId === view.me.userId;

  async function open() {
    const r = await socket.command(view.world.id, { type: "startWorld" });
    if (!r.ok) setError(r.reason);
  }

  return (
    <main className="founding-frame founding-frame--open">
      <Module weight="heavy" className="open-bench">
        {company && (
          <div className="flex flex-col items-center gap-4">
            {/* The new company's badge is stamped onto the bench (C61). */}
            <div className="stamp-in">
              <EmblemBadge
                logo={company.logo}
                palette={company.palette}
                motion={company.logoMotion}
                size={220}
                label={company.name}
              />
            </div>
            <h1 className="player-text m-0 text-center font-display text-4xl font-bold text-accent">
              {company.name}
            </h1>
          </div>
        )}
        <p className="max-w-md text-center text-muted">
          You start with your founding cash and no AI yet. Once the world opens, its clock starts
          and you can train your first AI.
        </p>
        {view.world.phase === "setup" &&
          (isHost ? (
            <PushButton className="open-bench__switch" onClick={open}>
              Open for business
            </PushButton>
          ) : (
            <p className="flex items-center gap-2 text-muted">
              <Lamp state="wait" label="Waiting" blink />
              Waiting for the host to open the world…
            </p>
          ))}
        {error && (
          <p role="alert" className="text-sm text-danger">
            {error}
          </p>
        )}
      </Module>
      <div className="open-side">
        <Players view={view} />
        <InviteFriends code={view.world.inviteCode} />
      </div>
    </main>
  );
}
