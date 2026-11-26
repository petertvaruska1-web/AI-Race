import type { ReactNode } from "react";

/**
 * A recessed dark display (design §5.1), always dark on either glass. With `scroll`, long
 * content scrolls inside it, and it can be reached with the keyboard and is named by `label`.
 */
export function Display({
  scroll = false,
  label,
  className = "",
  children,
}: {
  scroll?: boolean;
  label?: string;
  className?: string;
  children: ReactNode;
}) {
  if (scroll) {
    return (
      <section
        aria-label={label}
        // biome-ignore lint/a11y/noNoninteractiveTabindex: a scrolling display must be reachable by keyboard
        tabIndex={0}
        className={`display display--scroll ${className}`}
      >
        {children}
      </section>
    );
  }
  return <div className={`display ${className}`}>{children}</div>;
}
