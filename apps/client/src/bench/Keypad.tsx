/** A code typed or pasted: capitals and digits only, no spaces, at most `maxLength`. */
export function cleanCode(s: string, maxLength: number): string {
  return s
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "")
    .slice(0, maxLength);
}

/**
 * A keypad for a code (the lab door's join keypad, C62): a display to type or paste into, and a
 * key per character of `alphabet` plus Delete.
 */
export function Keypad({
  label,
  alphabet,
  value,
  maxLength,
  onChange,
  placeholder = "ABC123",
}: {
  label: string;
  alphabet: string;
  value: string;
  maxLength: number;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <div className="keypad">
      <input
        aria-label={label}
        className="display-input keypad__display"
        value={value}
        placeholder={placeholder}
        autoComplete="off"
        spellCheck={false}
        onChange={(e) => onChange(cleanCode(e.target.value, maxLength))}
      />
      <div className="keypad__keys">
        {[...alphabet].map((ch) => (
          <button
            key={ch}
            type="button"
            className="keypad__key"
            onClick={() => onChange(cleanCode(value + ch, maxLength))}
          >
            {ch}
          </button>
        ))}
        <button
          type="button"
          aria-label="Delete"
          className="keypad__key keypad__key--wide"
          onClick={() => onChange(value.slice(0, -1))}
        >
          ⌫
        </button>
      </div>
    </div>
  );
}
