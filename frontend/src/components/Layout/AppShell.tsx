import {
  Activity,
  Bell,
  Gauge,
  LayoutDashboard,
  LineChart,
  MoreHorizontal,
  Newspaper,
  Settings,
  WalletCards,
} from "lucide-react";
import type { ReactNode } from "react";

export type PageKey =
  | "home"
  | "market"
  | "screener"
  | "news"
  | "portfolio"
  | "alerts"
  | "signals"
  | "settings"
  | "more";

const desktopSections = [
  {
    label: "Visión general",
    items: [{ key: "home" as const, label: "Inicio", icon: LayoutDashboard }],
  },
  {
    label: "Mercados",
    items: [
      { key: "market" as const, label: "Economía", icon: Gauge },
      { key: "screener" as const, label: "Acciones y CEDEARs", icon: LineChart },
      { key: "news" as const, label: "Noticias", icon: Newspaper },
    ],
  },
  {
    label: "Gestión",
    items: [
      { key: "portfolio" as const, label: "Portfolio", icon: WalletCards },
      { key: "alerts" as const, label: "Alertas", icon: Bell },
      { key: "signals" as const, label: "Lecturas automáticas", icon: Activity },
    ],
  },
  {
    label: "Sistema",
    items: [{ key: "settings" as const, label: "Ajustes", icon: Settings }],
  },
];

const mobileNav = [
  { key: "home" as const, label: "Inicio", icon: LayoutDashboard },
  { key: "market" as const, label: "Mercado", icon: Gauge },
  { key: "screener" as const, label: "Screener", icon: LineChart },
  { key: "portfolio" as const, label: "Portfolio", icon: WalletCards },
  { key: "alerts" as const, label: "Alertas", icon: Bell },
  { key: "more" as const, label: "Más", icon: MoreHorizontal },
];

interface Props {
  active: PageKey;
  onNavigate: (page: PageKey) => void;
  children: ReactNode;
}

export function AppShell({ active, onNavigate, children }: Props) {
  return (
    <div className="min-h-screen bg-obs-bg text-obs-text">
      <aside className="fixed left-0 top-0 hidden h-full w-72 overflow-y-auto border-r border-obs-border bg-obs-card/95 p-5 lg:block">
        <div className="mb-7">
          <div className="text-xs uppercase tracking-wide text-obs-blue">Observatorio económico</div>
          <h1 className="mt-1 text-2xl font-semibold">ObeconBot</h1>
        </div>
        <nav className="space-y-5">
          {desktopSections.map((section) => (
            <div key={section.label}>
              <div className="mb-1 px-3 text-xs uppercase tracking-wide text-obs-muted">
                {section.label}
              </div>
              <div className="space-y-1">
                {section.items.map((item) => {
                  const Icon = item.icon;
                  const selected = active === item.key;
                  return (
                    <button
                      key={item.key}
                      type="button"
                      className={`flex min-h-11 w-full items-center gap-3 rounded-md border px-3 text-left text-sm transition ${
                        selected
                          ? "border-obs-blue bg-obs-card2 text-obs-text"
                          : "border-transparent text-obs-muted hover:border-obs-border hover:bg-obs-card2"
                      }`}
                      onClick={() => onNavigate(item.key)}
                    >
                      <Icon size={18} />
                      <span>{item.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>
      </aside>
      <main className="min-w-0 pb-24 lg:ml-72 lg:pb-0">
        <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">{children}</div>
      </main>
      <nav className="fixed bottom-0 left-0 right-0 z-20 grid grid-cols-6 border-t border-obs-border bg-obs-card/95 backdrop-blur lg:hidden">
        {mobileNav.map((item) => {
          const Icon = item.icon;
          const selected =
            active === item.key ||
            (item.key === "more" && ["news", "signals", "settings"].includes(active));
          return (
            <button
              key={item.key}
              type="button"
              className={`flex min-h-[64px] min-w-0 flex-col items-center justify-center gap-1 px-0.5 text-[10px] ${
                selected ? "text-obs-blue" : "text-obs-muted"
              }`}
              onClick={() => onNavigate(item.key)}
            >
              <Icon size={19} />
              <span className="max-w-full truncate">{item.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}
