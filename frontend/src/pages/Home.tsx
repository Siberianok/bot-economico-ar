import { ArrowRight, Bell, Newspaper, WalletCards } from "lucide-react";
import type { PageProps } from "../App";
import { getDailySummary } from "../api/summary";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { KpiCard } from "../components/Cards/KpiCard";
import { DataState } from "../components/States/DataState";
import { useEnvelope } from "../hooks/useEnvelope";

const money = new Intl.NumberFormat("es-AR", {
  style: "currency",
  currency: "ARS",
  maximumFractionDigits: 2,
});
const number = new Intl.NumberFormat("es-AR", { maximumFractionDigits: 2 });

function value(block: any, ...keys: string[]): number | null {
  for (const key of keys) {
    const candidate = block?.data?.[key];
    if (typeof candidate === "number" && Number.isFinite(candidate)) return candidate;
  }
  return null;
}

function ActionRow({
  icon: Icon,
  label,
  meta,
  onClick,
}: {
  icon: typeof Bell;
  label: string;
  meta: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="flex min-h-16 w-full items-center gap-3 border-b border-obs-border px-1 text-left last:border-b-0"
      onClick={onClick}
    >
      <span className="flex size-9 shrink-0 items-center justify-center rounded-md bg-obs-card2 text-obs-blue">
        <Icon size={18} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-medium">{label}</span>
        <span className="block truncate text-xs text-obs-muted">{meta}</span>
      </span>
      <ArrowRight size={17} className="shrink-0 text-obs-muted" />
    </button>
  );
}

export function Home({ onNavigate }: PageProps) {
  const { payload, loading, error } = useEnvelope(getDailySummary);
  if (loading) return <DataState title="Preparando resumen" detail="Actualizando mercado y noticias." />;
  if (error) return <DataState title="Error de carga" detail={error} />;
  if (!payload) return <DataState title="Resumen no disponible" />;

  const market = payload.data?.market || {};
  const news = payload.data?.news?.items || [];
  const ccl = value((market as any).dolar_ccl, "venta");
  const mep = value((market as any).dolar_mep, "venta");
  const risk = value((market as any).riesgo_pais, "value");
  const inflation = value((market as any).inflacion, "value");
  const readings = Array.isArray((market as any).signals?.data) ? (market as any).signals.data : [];

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-xs uppercase tracking-wide text-obs-blue">Resumen diario</div>
          <h2 className="text-3xl font-semibold">Observatorio económico</h2>
        </div>
        <div className="flex gap-2">
          <StatusBadge status={payload.status} />
          <StatusBadge status={payload.freshness} />
        </div>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard title="Dólar CCL" value={ccl == null ? "No disponible" : money.format(ccl)} tone="blue" />
        <KpiCard title="Dólar MEP" value={mep == null ? "No disponible" : money.format(mep)} />
        <KpiCard title="Riesgo país" value={risk == null ? "No disponible" : `${number.format(risk)} pb`} tone="warning" />
        <KpiCard title="Inflación mensual" value={inflation == null ? "No disponible" : `${number.format(inflation)}%`} />
      </div>

      <section className="grid gap-5 xl:grid-cols-[1.35fr_0.65fr]">
        <div className="min-w-0">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-lg font-semibold">Noticias destacadas</h3>
            <button type="button" className="text-sm text-obs-blue" onClick={() => onNavigate?.("news")}>Ver todas</button>
          </div>
          <div className="divide-y divide-obs-border border-y border-obs-border">
            {news.slice(0, 4).map((item) => (
              <a
                key={item.url}
                className="block py-4 hover:text-obs-blue"
                href={item.url}
                target="_blank"
                rel="noreferrer"
              >
                <div className="text-sm font-medium">{item.title}</div>
                <div className="mt-1 text-xs text-obs-muted">{item.source}</div>
              </a>
            ))}
            {!news.length ? <DataState title="Noticias no disponibles" /> : null}
          </div>
        </div>

        <div>
          <h3 className="mb-3 text-lg font-semibold">Accesos</h3>
          <div className="border-y border-obs-border">
            <ActionRow icon={WalletCards} label="Portfolio" meta="Resumen, composición y rendimiento" onClick={() => onNavigate?.("portfolio")} />
            <ActionRow icon={Bell} label="Alertas" meta="Reglas guardadas e historial" onClick={() => onNavigate?.("alerts")} />
            <ActionRow icon={Newspaper} label="Noticias" meta={`${news.length} notas disponibles`} onClick={() => onNavigate?.("news")} />
          </div>
        </div>
      </section>

      <section>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-semibold">Lecturas automáticas</h3>
          <button type="button" className="text-sm text-obs-blue" onClick={() => onNavigate?.("signals")}>Ver historial</button>
        </div>
        {readings.length ? (
          <div className="grid gap-3 md:grid-cols-2">
            {readings.slice(0, 4).map((reading: any) => (
              <div key={reading.dedup_key} className="border-l-2 border-obs-violet bg-obs-card px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium">{reading.title}</span>
                  <span className="text-xs text-obs-violet">{reading.score}/100</span>
                </div>
                <div className="mt-1 text-sm text-obs-muted">{reading.description}</div>
              </div>
            ))}
          </div>
        ) : (
          <DataState title="Sin lecturas activas" detail="Los indicadores observados no superan los umbrales configurados." />
        )}
      </section>
    </div>
  );
}
