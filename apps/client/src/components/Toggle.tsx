import { KeyRow } from "../bench/KeyRow.tsx";

/** A small row of illuminated keys for a choice (native radio buttons underneath). */
export function Toggle<T extends string>({
  legend,
  options,
  value,
  onChange,
}: {
  legend: string;
  options: { label: string; value: T }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <KeyRow legend={legend} options={options} value={value} onChange={onChange} className="mb-2" />
  );
}
