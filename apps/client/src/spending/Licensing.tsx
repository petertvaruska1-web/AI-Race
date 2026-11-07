import {
  type Command,
  GROUP_IDS,
  type GroupId,
  LICENSE_MONTHS,
  LICENSE_PRICE_CENTS,
  type LicenseOfferView,
  type WorldView,
} from "@airace/protocol";
import { groupList } from "@airace/sim";
import { useState } from "react";
import { Button } from "../components/Button.tsx";
import { INPUT } from "../components/Field.tsx";
import { Panel } from "../components/Panel.tsx";
import { dayLabel, formatMoney } from "../format.ts";
import { GROUPS } from "../groups.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, WindowFrame, type WindowProps } from "../windows/WindowFrame.tsx";

// Licensing (C44): rent your AIs to other companies, or theirs for your customers, for
// royalties per customer a month. Every number here comes from the world view.

/** "$2.50" from cents. */
export function centsLabel(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}

/** "$2.50 a customer a month for 6 months". */
export function termsLabel(t: { priceCents: number; months: number }): string {
  return `${centsLabel(t.priceCents)} a customer a month for ${t.months} months`;
}

const nameOf = (view: WorldView, companyId: string) =>
  view.companies.find((c) => c.id === companyId)?.name ?? "A company";

function useCommand(view: WorldView) {
  const socket = useSocket();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async (cmd: Command) => {
    setBusy(true);
    setError(null);
    const r = await socket.command(view.world.id, cmd);
    setBusy(false);
    if (!r.ok) setError(r.reason);
    return r.ok;
  };
  return { run, error, busy };
}

function Problem({ error }: { error: string | null }) {
  return error ? (
    <p role="alert" className="mt-2 text-sm text-danger">
      {error}
    </p>
  ) : null;
}

/** Checkboxes for customer groups, in their usual order. */
function GroupPicker({
  groups,
  chosen,
  onChange,
}: {
  groups: readonly GroupId[];
  chosen: GroupId[];
  onChange: (next: GroupId[]) => void;
}) {
  return (
    <fieldset className="mb-4">
      <legend className="mb-1 text-sm font-semibold">For these customers</legend>
      <div className="grid gap-1 sm:grid-cols-3">
        {groups.map((g) => (
          <label key={g} className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={chosen.includes(g)}
              onChange={(e) =>
                onChange(GROUP_IDS.filter((x) => (x === g ? e.target.checked : chosen.includes(x))))
              }
            />
            {GROUPS[g].many}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

/** Rent one of your AIs to any company in the world (from its AI window). */
export function OfferWindow({ entry, view, versionId }: WindowProps & { versionId: string }) {
  const company = view.me.company;
  const back = useWindows((s) => s.back);
  const { run, error, busy } = useCommand(view);
  const [price, setPrice] = useState("3.00");
  const [months, setMonths] = useState<(typeof LICENSE_MONTHS)[number]>(6);
  const [groups, setGroups] = useState<GroupId[]>([]);
  const [exclusive, setExclusive] = useState(false);
  const v = company?.versions.find((x) => x.id === versionId);
  const onSale = company?.products.find((p) => p.versionId === versionId && p.status === "live");
  if (!company || !v) return <GoneWindow entry={entry} />;
  const cents = Math.round(Number(price) * 100);
  return (
    <WindowFrame
      entry={entry}
      title={`Offer ${onSale?.name ?? `your AI ${v.label}`} for license`}
      subtitle="Other companies can use it for their customers and pay you for each one, every month. They never see how it was made."
    >
      {!onSale ? (
        <p className="text-sm text-muted">
          Put this AI on sale first: only AIs on sale are licensed.
        </p>
      ) : (
        <>
          <label className="mb-4 block text-sm">
            <span className="mb-1 block font-semibold">Price per customer a month ($)</span>
            <input
              className={INPUT}
              type="number"
              min={LICENSE_PRICE_CENTS.min / 100}
              max={LICENSE_PRICE_CENTS.max / 100}
              step={0.1}
              value={price}
              onChange={(e) => setPrice(e.target.value)}
            />
          </label>
          <label className="mb-4 block text-sm">
            <span className="mb-1 block font-semibold">How long</span>
            <select
              className={INPUT}
              value={months}
              onChange={(e) => setMonths(Number(e.target.value) as (typeof LICENSE_MONTHS)[number])}
            >
              {LICENSE_MONTHS.map((m) => (
                <option key={m} value={m}>{`${m} months`}</option>
              ))}
            </select>
          </label>
          <GroupPicker groups={GROUP_IDS} chosen={groups} onChange={setGroups} />
          <label className="mb-4 flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={exclusive}
              onChange={(e) => setExclusive(e.target.checked)}
            />
            Only one company may license it
          </label>
          <Button
            disabled={busy || groups.length === 0 || !Number.isFinite(cents)}
            onClick={async () => {
              const ok = await run({
                type: "offerLicense",
                versionId,
                priceCents: cents,
                months,
                groups,
                exclusive,
              });
              if (ok) back();
            }}
          >
            Offer it
          </Button>
        </>
      )}
      <Problem error={error} />
    </WindowFrame>
  );
}

/** Take an offer for one of your AIs on sale. */
export function TakeWindow({ entry, view, offerId }: WindowProps & { offerId: string }) {
  const company = view.me.company;
  const back = useWindows((s) => s.back);
  const { run, error, busy } = useCommand(view);
  const offer = view.licenseOffers.find((o) => o.id === offerId);
  const products = company?.products.filter((p) => p.status === "live") ?? [];
  const [productId, setProductId] = useState(products[0]?.id ?? "");
  const [groups, setGroups] = useState<GroupId[]>(offer ? [...offer.groups] : []);
  if (!company || !offer) return <GoneWindow entry={entry} />;
  const version = view.companies
    .find((c) => c.id === offer.licensorId)
    ?.products.find((p) => p.version?.id === offer.versionId)?.version;
  return (
    <WindowFrame
      entry={entry}
      title={`License ${offer.aiName}`}
      subtitle={`From ${nameOf(view, offer.licensorId)}: ${termsLabel(offer)}${offer.exclusive ? ", only for you" : ""}. It answers your customers in the groups you pick, and you pay for each of them at every month's end.`}
    >
      {version && (
        <p className="mb-4 text-sm">
          {`Overall score ${Math.round(version.overall ?? 0)}`}
          {version.personalityWords.length > 0 && ` · ${version.personalityWords.join(", ")}`}
        </p>
      )}
      {products.length === 0 ? (
        <p className="text-sm text-muted">
          Put an AI on sale first: a licensed AI powers one of yours.
        </p>
      ) : (
        <>
          <label className="mb-4 block text-sm">
            <span className="mb-1 block font-semibold">For your AI</span>
            <select
              className={INPUT}
              value={productId}
              onChange={(e) => setProductId(e.target.value)}
            >
              {products.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <GroupPicker groups={offer.groups} chosen={groups} onChange={setGroups} />
          <p className="mb-4 text-xs text-muted">
            Ending it early costs one more month of royalties.
          </p>
          <Button
            disabled={busy || groups.length === 0 || !productId}
            onClick={async () => {
              const ok = await run({ type: "takeLicense", offerId, productId, groups });
              if (ok) back();
            }}
          >
            License it
          </Button>
        </>
      )}
      <Problem error={error} />
    </WindowFrame>
  );
}

/** Ask another player to license their AI to you, on your terms. */
export function ProposeWindow({ entry, view, versionId }: WindowProps & { versionId: string }) {
  const company = view.me.company;
  const back = useWindows((s) => s.back);
  const { run, error, busy } = useCommand(view);
  const products = company?.products.filter((p) => p.status === "live") ?? [];
  const [productId, setProductId] = useState(products[0]?.id ?? "");
  const [price, setPrice] = useState("3.00");
  const [months, setMonths] = useState<(typeof LICENSE_MONTHS)[number]>(6);
  const [groups, setGroups] = useState<GroupId[]>([]);
  const maker = view.companies.find((c) =>
    c.products.some((p) => p.status === "live" && p.version?.id === versionId),
  );
  const product = maker?.products.find((p) => p.version?.id === versionId);
  if (!company || !maker || !product) return <GoneWindow entry={entry} />;
  const cents = Math.round(Number(price) * 100);
  return (
    <WindowFrame
      entry={entry}
      title={`Ask to license ${product.name}`}
      subtitle={`${maker.name} decides. If they accept, ${product.name} answers your customers in the groups you pick.`}
    >
      {products.length === 0 ? (
        <p className="text-sm text-muted">
          Put an AI on sale first: a licensed AI powers one of yours.
        </p>
      ) : (
        <>
          <label className="mb-4 block text-sm">
            <span className="mb-1 block font-semibold">For your AI</span>
            <select
              className={INPUT}
              value={productId}
              onChange={(e) => setProductId(e.target.value)}
            >
              {products.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <label className="mb-4 block text-sm">
            <span className="mb-1 block font-semibold">Price per customer a month ($)</span>
            <input
              className={INPUT}
              type="number"
              min={LICENSE_PRICE_CENTS.min / 100}
              max={LICENSE_PRICE_CENTS.max / 100}
              step={0.1}
              value={price}
              onChange={(e) => setPrice(e.target.value)}
            />
          </label>
          <label className="mb-4 block text-sm">
            <span className="mb-1 block font-semibold">How long</span>
            <select
              className={INPUT}
              value={months}
              onChange={(e) => setMonths(Number(e.target.value) as (typeof LICENSE_MONTHS)[number])}
            >
              {LICENSE_MONTHS.map((m) => (
                <option key={m} value={m}>{`${m} months`}</option>
              ))}
            </select>
          </label>
          <GroupPicker groups={GROUP_IDS} chosen={groups} onChange={setGroups} />
          <Button
            disabled={busy || groups.length === 0 || !productId}
            onClick={async () => {
              const ok = await run({
                type: "proposeLicense",
                versionId,
                productId,
                priceCents: cents,
                months,
                groups,
              });
              if (ok) back();
            }}
          >
            Ask them
          </Button>
        </>
      )}
      <Problem error={error} />
    </WindowFrame>
  );
}

/** A company asks to license your AI: accept or decline. */
export function ProposalWindow({ entry, view, proposalId }: WindowProps & { proposalId: string }) {
  const back = useWindows((s) => s.back);
  const { run, error, busy } = useCommand(view);
  const p = view.me.company?.licenses.proposals.find((x) => x.id === proposalId);
  if (!p) return <GoneWindow entry={entry} />;
  const asking = p.role === "licensor";
  const answer = async (accept: boolean) => {
    if (await run({ type: "answerProposal", proposalId, accept })) back();
  };
  return (
    <WindowFrame
      entry={entry}
      title={
        asking ? `${p.otherName} asks to license ${p.aiName}` : `You asked to license ${p.aiName}`
      }
      subtitle={`For ${p.productName}'s ${groupList(p.groups)}: ${termsLabel(p)}. They never see how your AI was made; its answers go to their customers.`}
    >
      <p className="text-sm">
        {asking
          ? `They pay at every month's end for each of their customers it answers. The request lapses on ${dayLabel(p.expiresDay)}.`
          : `${p.otherName} hasn't answered yet. The request lapses on ${dayLabel(p.expiresDay)}.`}
      </p>
      {asking && (
        <div className="mt-6 flex flex-wrap gap-3">
          <Button disabled={busy} onClick={() => answer(true)}>
            Accept
          </Button>
          <Button variant="secondary" disabled={busy} onClick={() => answer(false)}>
            Decline
          </Button>
        </div>
      )}
      <Problem error={error} />
    </WindowFrame>
  );
}

/** Offers anyone can take: each with "License it" (yours with "Withdraw"). */
export function OffersList({ view, offers }: { view: WorldView; offers: LicenseOfferView[] }) {
  const open = useWindows((s) => s.open);
  const mine = view.me.company?.id;
  const { run, error, busy } = useCommand(view);
  return (
    <>
      <ul className="divide-y divide-border">
        {offers.map((o) => (
          <li key={o.id} className="flex flex-wrap items-center gap-3 py-3">
            <p className="mr-auto text-sm">
              <span className="font-semibold">{o.aiName}</span>{" "}
              <span className="text-muted">
                {`${nameOf(view, o.licensorId)} · ${termsLabel(o)} · ${groupList(o.groups)}${o.exclusive ? " · only one licensee" : ""}`}
              </span>
            </p>
            {o.licensorId === mine ? (
              <Button
                variant="secondary"
                disabled={busy}
                onClick={() => run({ type: "withdrawOffer", offerId: o.id })}
              >
                Withdraw
              </Button>
            ) : (
              <Button
                aria-label={`License ${o.aiName}`}
                onClick={(e) => open({ kind: "takeLicense", offerId: o.id }, e.currentTarget)}
              >
                License it
              </Button>
            )}
          </li>
        ))}
      </ul>
      <Problem error={error} />
    </>
  );
}

/** Your licenses: AIs you rent out or rent, requests waiting, and your open offers. */
export function LicensesPanel({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const company = view.me.company;
  const { run, error, busy } = useCommand(view);
  const [ending, setEnding] = useState<string | null>(null);
  if (!company) return null;
  const { deals, proposals } = company.licenses;
  const offers = view.licenseOffers.filter((o) => o.licensorId === company.id);
  return (
    <Panel
      title="Licenses"
      aria-label="Licenses"
      hint="Rent your AIs to other companies, or theirs for your customers. Royalties are paid for each customer at every month's end."
    >
      {deals.length + proposals.length + offers.length === 0 ? (
        <p className="text-sm text-muted">
          No licenses yet. Offer one of your AIs from its window, or license one from the AI
          Directory.
        </p>
      ) : (
        <ul className="space-y-3">
          {deals.map((d) => (
            <li key={d.id} className="rounded-xl border border-border p-3 text-sm">
              <p className="font-semibold">{`${d.aiName} powers ${d.productName} for ${groupList(d.groups)}`}</p>
              <p className="text-xs text-muted">
                {d.role === "licensee"
                  ? `From ${d.otherName} · ${termsLabel({ priceCents: d.priceCents, months: Math.round((d.endDay - d.startDay) / 30) })} · until ${dayLabel(d.endDay)} · ${formatMoney(d.owed)} owed so far this month`
                  : `For ${d.otherName} · ${centsLabel(d.priceCents)} a customer a month · until ${dayLabel(d.endDay)} · ${formatMoney(d.owed)} owed so far this month`}
              </p>
              {d.role === "licensee" &&
                (ending === d.id ? (
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <span className="text-xs">It costs one more month of royalties.</span>
                    <Button
                      disabled={busy}
                      onClick={async () => {
                        if (await run({ type: "endLicense", dealId: d.id })) setEnding(null);
                      }}
                    >
                      Yes, end it
                    </Button>
                    <Button variant="secondary" onClick={() => setEnding(null)}>
                      Keep it
                    </Button>
                  </div>
                ) : (
                  <Button className="mt-2" variant="secondary" onClick={() => setEnding(d.id)}>
                    End early
                  </Button>
                ))}
            </li>
          ))}
          {proposals.map((p) => (
            <li key={p.id}>
              <button
                type="button"
                className="row-btn"
                onClick={(e) => open({ kind: "proposal", proposalId: p.id }, e.currentTarget)}
              >
                <span>
                  {p.role === "licensor"
                    ? `${p.otherName} asks to license ${p.aiName}`
                    : `You asked ${p.otherName} for ${p.aiName}`}
                </span>
                <span aria-hidden="true" className="ml-auto text-muted">
                  ›
                </span>
              </button>
            </li>
          ))}
          {offers.length > 0 && (
            <li>
              <p className="text-xs font-semibold uppercase tracking-wider text-muted">
                Your offers
              </p>
              <OffersList view={view} offers={offers} />
            </li>
          )}
        </ul>
      )}
      <Problem error={error} />
    </Panel>
  );
}
