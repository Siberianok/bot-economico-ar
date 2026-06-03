import { getPortfolioSummary } from "../api/portfolio";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { KpiCard } from "../components/Cards/KpiCard";
import { DataState } from "../components/States/DataState";
import { useEnvelope } from "../hooks/useEnvelope";

export function PortfolioCommandCenter() {
  const { payload, loading, error } = useEnvelope(getPortfolioSummary);
  const data = payload?.data || {};

  return (
    <div className="space-y-5">
      <header>
        <div className="text-xs uppercase tracking-wide text-obs-blue">Portfolio</div>
        <h2 className="text-3xl font-semibold">Command Center</h2>
      </header>
      {loading ? <DataState title="Cargando portfolio" /> : null}
      {error ? <DataState title="Error de carga" detail={error} /> : null}
      {payload ? (
        <div className="flex gap-2">
          <StatusBadge status={payload.status} />
          <StatusBadge status={payload.freshness} />
        </div>
      ) : null}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <KpiCard title="Valor invertido" value={(data as any).invested_value?.data ?? "No disponible"} />
        <KpiCard title="Valor actual" value={(data as any).current_value?.data ?? "No disponible"} />
        <KpiCard title="TWR" value={(data as any).twr?.data ?? "No disponible"} />
        <KpiCard title="MWR" value={(data as any).mwr?.data ?? "No disponible"} />
      </div>
      <section className="grid gap-4 lg:grid-cols-3">
        <DataState title="Composición" detail={(data as any).composition?.status || "not_available"} />
        <DataState title="Exposición por moneda" detail={(data as any).currency_exposure?.status || "not_available"} />
        <DataState title="Rebalanceo" detail={(data as any).rebalance?.status || "not_available"} />
      </section>
      {payload?.warnings.length ? <DataState title="Advertencias" detail={payload.warnings.join(" · ")} /> : null}
    </div>
  );
}

