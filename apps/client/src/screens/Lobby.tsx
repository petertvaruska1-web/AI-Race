import { type Logo as LogoSpec, type Palette, TEXT_LIMITS, type WorldView } from "@airace/protocol";
import { PALETTE_PRESETS } from "@airace/sim";
import { type FormEvent, useState } from "react";
import { Button } from "../components/Button.tsx";
import { Field, INPUT } from "../components/Field.tsx";
import { Logo } from "../components/Logo.tsx";
import { LogoBuilder } from "../components/LogoBuilder.tsx";
import { PalettePicker } from "../components/PalettePicker.tsx";
import { Panel } from "../components/Panel.tsx";
import { useSocket } from "../store/session.ts";

const FIRST_PRESET = PALETTE_PRESETS[0]?.palette ?? { primary: "#4f8cff", secondary: "#38d9c0" };

/** Founding a company (C5: its palette becomes the player's whole dashboard), then opening the world. */
export function Lobby({ view }: { view: WorldView }) {
  return view.me.company ? <OpenForBusiness view={view} /> : <FoundCompany view={view} />;
}

function FoundCompany({ view }: { view: WorldView }) {
  const socket = useSocket();
  const [name, setName] = useState("");
  const [logo, setLogo] = useState<LogoSpec>({ shape: "circle", symbol: "spark", initials: "" });
  const [palette, setPalette] = useState<Palette>(FIRST_PRESET);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [min, max] = TEXT_LIMITS.companyName;

  async function found(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const r = await socket.command(view.world.id, { type: "createCompany", name, logo, palette });
    setBusy(false);
    if (!r.ok) setError(r.reason);
  }

  return (
    <main className="mx-auto max-w-4xl space-y-6 px-6 py-10">
      <header>
        <p className="text-sm text-muted">{view.world.name}</p>
        <h1 className="text-3xl font-bold">Found your company</h1>
        <p className="text-muted">
          Its colors become your whole dashboard. You can't change them later.
        </p>
      </header>
      <form onSubmit={found} className="grid gap-6 md:grid-cols-[1fr_280px]">
        <div className="space-y-6">
          <Panel>
            <Field label="Company name" hint={`${min}–${max} characters.`}>
              <input
                className={INPUT}
                value={name}
                maxLength={max}
                onChange={(e) => setName(e.target.value)}
                placeholder="Acme Intelligence"
              />
            </Field>
          </Panel>
          <Panel title="Logo">
            <LogoBuilder logo={logo} palette={palette} onChange={setLogo} />
          </Panel>
          <Panel title="Colors">
            <PalettePicker value={palette} onChange={setPalette} />
          </Panel>
        </div>
        <aside className="space-y-4">
          <Panel title="Preview" className="sticky top-6">
            <div
              data-testid="preview"
              className="flex flex-col items-center gap-3 rounded-xl p-6"
              style={{ background: "var(--bg)", borderTop: `4px solid ${palette.primary}` }}
            >
              <Logo logo={logo} palette={palette} size={96} title="Your logo" />
              <p
                className="player-text text-center text-lg font-semibold"
                style={{ color: palette.primary }}
              >
                {name || "Your company"}
              </p>
            </div>
            {error && (
              <p role="alert" className="mt-3 text-sm text-danger">
                {error}
              </p>
            )}
            <Button
              type="submit"
              className="mt-4 w-full"
              disabled={busy || name.trim().length < min}
            >
              Found company
            </Button>
          </Panel>
        </aside>
      </form>
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
    <main className="mx-auto mt-16 max-w-lg space-y-6 px-6 text-center">
      {company && (
        <div className="flex flex-col items-center gap-3">
          <Logo logo={company.logo} palette={company.palette} size={112} title={company.name} />
          <h1 className="player-text text-3xl font-bold text-brand">{company.name}</h1>
        </div>
      )}
      <p className="text-muted">
        You start with your founding cash and no AI yet. Once the world opens, its clock starts and
        you can train your first AI.
      </p>
      {view.world.phase === "setup" &&
        (isHost ? (
          <Button onClick={open}>Open for business</Button>
        ) : (
          <p className="text-muted">Waiting for the host to open the world…</p>
        ))}
      {error && (
        <p role="alert" className="text-sm text-danger">
          {error}
        </p>
      )}
    </main>
  );
}
