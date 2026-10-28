/** A face for a satisfaction (0..1): happy, neutral or unhappy, in green, grey or red. */
export function Face({ sat, size = 28 }: { sat: number; size?: number }) {
  const mood = sat >= 0.65 ? "Happy" : sat <= 0.35 ? "Unhappy" : "Neutral";
  const mouth =
    mood === "Happy"
      ? "M8 14.5c1.2 1.4 2.5 2 4 2s2.8-.6 4-2"
      : mood === "Unhappy"
        ? "M8 16.5c1.2-1.4 2.5-2 4-2s2.8.6 4 2"
        : "M8.5 15.5h7";
  const color =
    mood === "Happy" ? "var(--good)" : mood === "Unhappy" ? "var(--danger)" : "var(--muted)";
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      role="img"
      aria-label={mood}
      style={{ color }}
    >
      <circle cx="12" cy="12" r="9.5" fill="none" stroke="currentColor" strokeWidth={1.6} />
      <circle cx="9" cy="10" r="1.1" fill="currentColor" />
      <circle cx="15" cy="10" r="1.1" fill="currentColor" />
      <path d={mouth} fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" />
    </svg>
  );
}
