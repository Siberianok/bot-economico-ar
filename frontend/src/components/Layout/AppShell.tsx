import { Activity, Bell, Gauge, LineChart, Settings, WalletCards } from "lucide-react";
import type { ReactNode } from "react";

export type PageKey = "market" | "screener" | "portfolio" | "alerts" | "signals" | "settings";

const nav = [
  { key: "market" as const, label: "Mercado", icon: Gauge },
  { key: "screener" as const, label: "Screener", icon: LineChart },
  { key: "portfolio" as const, label: "Portfolio", icon: WalletCards },
  { key: "alerts" as const, label: "Alertas", icon: Bell },
  { key: "signals" as const, label: "Señales", icon: Activity },
  { key: "settings" as const, label: "Ajustes", icon: Settings },
];

interface Props {
  active: PageKey;
  onNavigate: (page: PageKey) => void;
  children: ReactNode;
}

export function AppShell({ active, onNavigate, children }: Props) {
  return (
    <div className="min-h-screen bg-obs-bg text-obs-text">
      <aside className="fixed left-0 top-0 hidden h-full w-72 border-r border-obs-border bg-obs-card/95 p-5 lg:block">
        <div className="mb-8">
          <div className="text-xs uppercase tracking-wide text-obs-blue">Observatorio</div>
          <h1 className="mt-1 text-2xl font-semibold">Económico</h1>
        </div>
        <nav className="space-y-2">
          {nav.map((item) => {
            const Icon = item.icon;
            const selected = active === item.key;
            return (
              <button
                key={item.key}
                className={`flex w-full items-center gap-3 rounded-md border px-3 py-3 text-left text-sm transition ${
                  selected
                    ? "border-obs-blue bg-obs-card2 text-obs-text"
                    : "border-transparent text-obs-muted hover:border-obs-border hover:bg-obs-card2"
                }`}
                onClick={() => onNavigate(item.key)}
              >
                <Icon size={18} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </aside>
      <main className="pb-24 lg:ml-72 lg:pb-0">
        <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">{children}</div>
      </main>
      <nav className="fixed bottom-0 left-0 right-0 z-20 grid grid-cols-6 border-t border-obs-border bg-obs-card/95 backdrop-blur lg:hidden">
        {nav.map((item) => {
          const Icon = item.icon;
          const selected = active === item.key;
          return (
            <button
              key={item.key}
              className={`flex min-h-[64px] flex-col items-center justify-center gap-1 text-[11px] ${
                selected ? "text-obs-blue" : "text-obs-muted"
              }`}
              onClick={() => onNavigate(item.key)}
            >
              <Icon size={19} />
              <span className="truncate">{item.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}

