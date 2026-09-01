import { lazy, Suspense, useEffect, useState, type ComponentType } from "react";
import { AppShell, type PageKey } from "./components/Layout/AppShell";
import { DataState } from "./components/States/DataState";

export interface PageProps {
  onNavigate?: (page: PageKey) => void;
}

const pages: Record<PageKey, ComponentType<PageProps>> = {
  home: lazy(() => import("./pages/Home").then((module) => ({ default: module.Home }))),
  market: lazy(() => import("./pages/MarketPulse").then((module) => ({ default: module.MarketPulse }))),
  screener: lazy(() => import("./pages/Screener").then((module) => ({ default: module.Screener }))),
  news: lazy(() => import("./pages/News").then((module) => ({ default: module.News }))),
  portfolio: lazy(() =>
    import("./pages/PortfolioCommandCenter").then((module) => ({
      default: module.PortfolioCommandCenter,
    })),
  ),
  alerts: lazy(() => import("./pages/Alerts").then((module) => ({ default: module.Alerts }))),
  signals: lazy(() => import("./pages/Signals").then((module) => ({ default: module.Signals }))),
  settings: lazy(() => import("./pages/Settings").then((module) => ({ default: module.Settings }))),
  more: lazy(() => import("./pages/More").then((module) => ({ default: module.More }))),
};

const pageKeys = new Set<PageKey>(Object.keys(pages) as PageKey[]);

function pageFromHash(): PageKey {
  const candidate = window.location.hash.replace(/^#\/?/, "") as PageKey;
  return pageKeys.has(candidate) ? candidate : "home";
}

export default function App() {
  const [active, setActive] = useState<PageKey>(pageFromHash);

  useEffect(() => {
    const onHashChange = () => setActive(pageFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  function navigate(page: PageKey) {
    window.location.hash = page;
    setActive(page);
  }

  const ActivePage = pages[active];

  return (
    <AppShell active={active} onNavigate={navigate}>
      <Suspense
        fallback={<DataState title="Cargando módulo" detail="Preparando la vista seleccionada." />}
      >
        <ActivePage onNavigate={navigate} />
      </Suspense>
    </AppShell>
  );
}
