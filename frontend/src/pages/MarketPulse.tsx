import { getMarketPulse } from "../api/market";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { KpiCard } from "../components/Cards/KpiCard";
import { DarkChart } from "../components/Charts/DarkChart";
import { DataState } from "../components/States/DataState";
import { DataTable } from "../components/Tables/DataTable";
import { useEnvelope } from "../hooks/useEnvelope";

function valueFromBlock(block: any): string {
  const data = block?.data;
  if (!data || typeof data !== "object") return "No disponible";
  const venta = data.venta ?? data.value ?? data.val;
  return venta == null ? "No disponible" : String(venta);
}

export function MarketPulse() {
  const { payload, loading, error } = useEnvelope(getMarketPulse);

  if (loading) return <DataState title="Cargando Market Pulse" detail="Consultando backend..." />;
  if (error) return <DataState title="Error de carga" detail={error} />;
  if (!payload) return <DataState title="Sin respuesta" />;

  const data = payload.data;
  const fxRows = ["dolar_oficial", "dolar_blue", "dolar_mep", "dolar_ccl", "dolar_cripto"].map((key) => [
    key.replace("dolar_", "").toUpperCase(),
    valueFromBlock((data as any)?.[key]),
    <StatusBadge status={(data as any)?.[key]?.status || "not_available"} />,
  ]);
  const chartValues = fxRows
    .map((row) => Number(String(row[1]).replace(",", ".")))
    .filter((value) => Number.isFinite(value));

  return (
    <div className="space-y-5">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-xs uppercase tracking-wide text-obs-blue">Market Pulse</div>
          <h2 className="text-3xl font-semibold">Variables macro y mercado</h2>
        </div>
        <div className="flex gap-2">
          <StatusBadge status={payload.status} />
          <StatusBadge status={payload.freshness} />
        </div>
      </header>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <KpiCard title="Dólar cripto" value={valueFromBlock((data as any)?.dolar_cripto)} tone="blue" />
        <KpiCard title="Dólar MEP" value={valueFromBlock((data as any)?.dolar_mep)} />
        <KpiCard title="Riesgo país" value={valueFromBlock((data as any)?.riesgo_pais)} tone="warning" />
        <KpiCard title="Reservas" value={valueFromBlock((data as any)?.reservas)} />
        <KpiCard title="Inflación" value={valueFromBlock((data as any)?.inflacion)} />
      </div>

      <section className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-lg border border-obs-border bg-obs-card p-4">
          <h3 className="mb-3 text-lg font-semibold">Tipos de cambio</h3>
          <DataTable headers={["Tipo", "Valor", "Estado"]} rows={fxRows} />
        </div>
        <div className="rounded-lg border border-obs-border bg-obs-card p-4">
          {chartValues.length ? (
            <DarkChart title="Valores disponibles" labels={fxRows.map((row) => String(row[0]))} values={chartValues} />
          ) : (
            <DataState title="Gráfico no disponible" detail="No hay datos reales suficientes para graficar." />
          )}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <DataState title="Bandas cambiarias" detail={(data as any)?.bandas_cambiarias?.status || "not_available"} />
        <DataState title="Señales macro" detail={(data as any)?.signals?.status || "not_available"} />
        <DataState title="Noticias / calendario" detail="Fuente pendiente de adaptar." />
      </section>
      {payload.warnings.length ? <DataState title="Advertencias" detail={payload.warnings.join(" · ")} /> : null}
    </div>
  );
}

