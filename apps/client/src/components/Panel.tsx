import type { HTMLAttributes, ReactNode } from "react";
import { Module } from "../bench/Module.tsx";

/** The game's basic surface: a bench module with an optional title, hint and actions. */
export function Panel({
  title,
  hint,
  actions,
  children,
  className = "",
  ...props
}: Omit<HTMLAttributes<HTMLElement>, "title"> & {
  title?: ReactNode;
  hint?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <Module label={title} hint={hint} actions={actions} className={className} {...props}>
      {children}
    </Module>
  );
}
