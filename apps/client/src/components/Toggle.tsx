/** A small row of pill-shaped choices (native radio buttons underneath). */
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
    <fieldset className="mb-2 flex gap-1 text-xs">
      <legend className="sr-only">{legend}</legend>
      {options.map((o) => (
        <label
          key={o.value}
          className={`cursor-pointer rounded-full px-3 py-1 focus-within:ring-2 focus-within:ring-brand ${value === o.value ? "bg-brand text-on-brand" : "bg-surface-2 text-muted"}`}
        >
          <input
            type="radio"
            name={legend}
            className="sr-only"
            checked={value === o.value}
            onChange={() => onChange(o.value)}
          />
          {o.label}
        </label>
      ))}
    </fieldset>
  );
}
