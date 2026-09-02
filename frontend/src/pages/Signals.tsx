import { getSignals } from "../api/signals";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { DataState } from "../components/States/DataState";
import { useEnvelope } from "../hooks/useEnvelope";
import { formatObservationDate } from "../lib/format";

const metricLabels: Record<string, string> = {
  gap_vs_official_pct: "Brecha vs. oficial",
};

const sourceLabels: Record<string, string> = {
  legacy_adapter: "Datos observados",
};

function signalValue(value: number | null | undefined, metric: string | null | undefined): string {
  if (value == null) return "-";
  const formatted = new Intl.NumberFormat("es-AR", { maximumFractionDigits: 2 }).format(value);
  return metric?.endsWith("_pct") ? `${formatted}%` : formatted;
}

export function Signals() {
  const { payload, loading, error } = useEnvelope(getSignals);
  const items = payload?.data?.items || [];

  return (
    <div className="space-y-5">
      <header>
        <div className="text-xs uppercase tracking-wide text-obs-blue">Monitoreo</div>
        <h2 className="text-3xl font-semibold">Lecturas automáticas</h2>
      </header>
      {loading ? <DataState title="Cargando señales" /> : null}
      {error ? <DataState title="Error de carga" detail={error} /> : null}
      <div className="grid gap-4">
        {items.map((signal) => (
          <article key={signal.id} className="rounded-lg border border-obs-border bg-obs-card p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-lg font-semibold">{signal.title}</h3>
              <div className="flex gap-2">
                <StatusBadge status={signal.severity} />
                <span className="rounded-full border border-obs-violet/50 bg-obs-violet/10 px-2 py-1 text-xs text-obs-violet">
                  {signal.score}/100
                </span>
              </div>
            </div>
            <p className="mt-2 text-sm text-obs-muted">{signal.description}</p>
            <div className="mt-4 grid gap-2 border-t border-obs-border pt-3 text-xs text-obs-muted sm:grid-cols-4">
              <div><span className="block uppercase">Activo</span><strong className="text-obs-text">{signal.asset || "-"}</strong></div>
              <div><span className="block uppercase">Métrica</span><strong className="text-obs-text">{metricLabels[signal.metric || ""] || signal.metric || signal.type}</strong></div>
              <div><span className="block uppercase">Valor</span><strong className="text-obs-text">{signalValue(signal.current_value, signal.metric)}</strong></div>
              <div><span className="block uppercase">Fuente</span><strong className="text-obs-text">{sourceLabels[signal.source] || signal.source}</strong></div>
            </div>
            <div className="mt-3 text-xs text-obs-muted">{formatObservationDate(signal.timestamp) || "Sin fecha"}</div>
          </article>
        ))}
        {!loading && !items.length ? <DataState title="Sin lecturas activas" detail="Los datos observados no superan los umbrales configurados." /> : null}
      </div>
      {payload?.warnings.length ? <DataState title="Advertencias" detail={payload.warnings.join(" · ")} /> : null}
    </div>
  );
}
