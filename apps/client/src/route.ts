import { useEffect, useState } from "react";

const TABS = ["company", "market", "money", "lab", "play", "bench"] as const;
export type Tab = (typeof TABS)[number];

/** Screens are addressed by the URL hash, so a reload returns to the same screen. */
export type Route = { screen: "home" } | { screen: "world"; worldId: string; tab: Tab };

/** Parses `#/w/<world>/<tab>`; the old `dashboard` (and anything unknown) opens the company. */
export function parseRoute(hash: string): Route {
  const [, w, worldId, tab] = hash.replace(/^#/, "").split("/");
  if (w === "w" && worldId) {
    const t = TABS.find((x) => x === tab) ?? "company";
    return { screen: "world", worldId, tab: t };
  }
  return { screen: "home" };
}

export function href(route: Route): string {
  return route.screen === "home" ? "#/" : `#/w/${route.worldId}/${route.tab}`;
}

export function go(route: Route): void {
  location.hash = href(route);
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseRoute(location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
