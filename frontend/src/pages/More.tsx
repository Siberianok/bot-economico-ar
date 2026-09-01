import { Activity, BarChart3, BellRing, ChevronRight, Newspaper, Settings } from "lucide-react";
import type { PageProps } from "../App";
import type { PageKey } from "../components/Layout/AppShell";
import { StatusBadge } from "../components/Badges/StatusBadge";

const miniappItems: Array<{ label: string; meta: string; page: PageKey; icon: typeof Activity }> = [
  { label: "Noticias", meta: "Fuentes RSS directas", page: "news", icon: Newspaper },
  { label: "Lecturas automáticas", meta: "Señales sobre datos observados", page: "signals", icon: Activity },
  { label: "Ajustes", meta: "API, Telegram y preferencias", page: "settings", icon: Settings },
];

const telegramItems = [
  { label: "Resumen diario", command: "/resumen", icon: BarChart3 },
  { label: "Suscripción diaria", command: "/subs", icon: BellRing },
  { label: "Performance operativa", command: "/performance", icon: Activity },
];

export function More({ onNavigate }: PageProps) {
  return (
    <div className="space-y-6">
      <header>
        <div className="text-xs uppercase tracking-wide text-obs-blue">Más</div>
        <h2 className="text-3xl font-semibold">Funciones del observatorio</h2>
      </header>

      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-obs-muted">Mini App</h3>
        <div className="divide-y divide-obs-border border-y border-obs-border">
          {miniappItems.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.page} type="button" className="flex min-h-16 w-full items-center gap-3 text-left" onClick={() => onNavigate?.(item.page)}>
                <Icon size={19} className="shrink-0 text-obs-blue" />
                <span className="min-w-0 flex-1"><span className="block font-medium">{item.label}</span><span className="block truncate text-xs text-obs-muted">{item.meta}</span></span>
                <ChevronRight size={18} className="text-obs-muted" />
              </button>
            );
          })}
        </div>
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-obs-muted">Telegram</h3>
        <div className="divide-y divide-obs-border border-y border-obs-border">
          {telegramItems.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.command} className="flex min-h-16 items-center gap-3">
                <Icon size={19} className="shrink-0 text-obs-muted" />
                <div className="min-w-0 flex-1"><div className="font-medium">{item.label}</div><div className="text-xs text-obs-muted">{item.command}</div></div>
                <StatusBadge status="telegram" />
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
