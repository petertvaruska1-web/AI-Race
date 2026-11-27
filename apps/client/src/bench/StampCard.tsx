import type { MouseEvent } from "react";

/**
 * A stamped card (design §10, Money's report cards): a small plate with its grade stamped in the
 * grade's color and a caption under it. A button named `label` that opens its story.
 */
export function StampCard({
  grade,
  caption,
  color,
  label,
  onClick,
}: {
  grade: string;
  caption: string;
  color: string;
  label: string;
  onClick: (e: MouseEvent<HTMLButtonElement>) => void;
}) {
  return (
    <button type="button" aria-label={label} className="stamp-card" onClick={onClick}>
      <span className="stamp-card__grade" style={{ color, borderColor: color }}>
        {grade}
      </span>
      <span className="stamp-card__caption">{caption}</span>
    </button>
  );
}
