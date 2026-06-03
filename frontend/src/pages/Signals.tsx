import { getSignals } from "../api/signals";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { DataState } from "../components/States/DataState";
import { useEnvelope } from "../hooks/useEnvelope";

export function Signals() {
  const { payload, loading, error } = useEnvelope(getSignals);
  const items = payload?.data?.items || [];

  return (
    <div className="space-y-5">
      <header>
        <div className="text-xs uppercase tracking-wide text-obs-blue">Señales</div>
        <h2 className="text-3xl font-semibold">Eventos automáticos</h2>
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
            <div className="mt-3 text-xs text-obs-muted">{signal.timestamp}</div>
          </article>
        ))}
        {!loading && !items.length ? <DataState title="Sin señales" detail="El motor inicial no generó señales nuevas." /> : null}
      </div>
      {payload?.warnings.length ? <DataState title="Advertencias" detail={payload.warnings.join(" · ")} /> : null}
    </div>
  );
}

