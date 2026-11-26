/** A quick-color cartridge (the founding bench, C61): a key showing two colors, named by its preset. */
export function Cartridge({
  label,
  primary,
  secondary,
  pressed,
  onClick,
}: {
  label: string;
  primary: string;
  secondary: string;
  pressed: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      aria-pressed={pressed}
      title={label}
      onClick={onClick}
      className="cartridge"
      style={{ background: `linear-gradient(90deg, ${primary} 50%, ${secondary} 50%)` }}
    />
  );
}
