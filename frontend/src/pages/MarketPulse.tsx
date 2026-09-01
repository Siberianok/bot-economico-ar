import { lazy, Suspense } from "react";
import type { PageProps } from "../App";
import { getMarketPulse } from "../api/market";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { KpiCard } from "../components/Cards/KpiCard";
import { DataState } from "../components/States/DataState";
import { DataTable } from "../components/Tables/DataTable";
import { useEnvelope } from "../hooks/useEnvelope";

const DarkChart = lazy(() =>
  import("../components/Charts/DarkChart").then((module) => ({ default: module.DarkChart })),
);

const numberFormatter = new Intl.NumberFormat("es-AR", { maximumFractionDigits: 2 });
const moneyFormatter = new Intl.NumberFormat("es-AR", {
  style: "currency",
  currency: "ARS",
  maximumFractionDigits: 2,
});

function numberFromData(data: any, ...keys: string[]): number | null {
  if (!data || typeof data !== "object") return null;
  for (const key of keys) {
    const value = data[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function fxValue(block: any, key: "compra" | "venta"): string {
  const value = numberFromData(block?.data, key);
  return value == null ? "No disponible" : moneyFormatter.format(value);
}

function percent(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${numberFormatter.format(value)}%`
    : "-";
}

function macroValue(block: any, suffix = ""): string {
  const value = numberFromData(block?.data, "value", "val");
  return value == null ? "No disponible" : `${numberFormatter.format(value)}${suffix}`;
}

function numericValueFromBlock(block: any): number | null {
  const data = block?.data;
  if (!data || typeof data !== "object") return null;
  const raw = data.venta ?? data.value ?? data.val;
  if (typeof raw === "number") return Number.isFinite(raw) ? raw : null;
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  const normalized = trimmed.includes(",")
    ? trimmed.replace(/\./g, "").replace(",", ".")
    : trimmed;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

export function MarketPulse({ onNavigate }: PageProps) {
  const { payload, loading, error } = useEnvelope(getMarketPulse);

  if (loading) return <DataState title="Cargando Market Pulse" detail="Consultando backend..." />;
  if (error) return <DataState title="Error de carga" detail={error} />;
  if (!payload) return <DataState title="Sin respuesta" />;

  const data = payload.data;
  const fxEntries = ["dolar_oficial", "dolar_blue", "dolar_mep", "dolar_ccl", "dolar_cripto"].map(
    (key) => ({
      label: key.replace("dolar_", "").toUpperCase(),
      block: (data as any)?.[key],
    }),
  );
  const fxRows = fxEntries.map(({ label, block }) => [
    label,
    fxValue(block, "compra"),
    fxValue(block, "venta"),
    percent(block?.data?.variation),
    block?.data?.fuente || "-",
    <StatusBadge status={block?.status || "not_available"} />,
  ]);
  const chartEntries = fxEntries
    .map(({ label, block }) => ({ label, value: numericValueFromBlock(block) }))
    .filter((entry): entry is { label: string; value: number } => entry.value !== null);
  const bands = (data as any)?.bandas_cambiarias?.data;
  const gaps = (data as any)?.brechas?.data;
  const readings = Array.isArray((data as any)?.signals?.data) ? (data as any).signals.data : [];

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
        <KpiCard
          title="Dólar cripto"
          value={fxValue((data as any)?.dolar_cripto, "venta")}
          meta={(data as any)?.dolar_cripto?.data?.fecha || undefined}
          tone="blue"
        />
        <KpiCard
          title="Dólar MEP"
          value={fxValue((data as any)?.dolar_mep, "venta")}
          meta={(data as any)?.dolar_mep?.data?.fecha || undefined}
        />
        <KpiCard
          title="Riesgo país"
          value={macroValue((data as any)?.riesgo_pais, " pb")}
          meta={(data as any)?.riesgo_pais?.data?.as_of || undefined}
          tone="warning"
        />
        <KpiCard
          title="Reservas"
          value={macroValue((data as any)?.reservas, " M USD")}
          meta={(data as any)?.reservas?.data?.as_of || undefined}
        />
        <KpiCard
          title="Inflación mensual"
          value={macroValue((data as any)?.inflacion, "%")}
          meta={(data as any)?.inflacion?.data?.as_of || undefined}
        />
      </div>

      <section className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="min-w-0 rounded-lg border border-obs-border bg-obs-card p-4">
          <h3 className="mb-3 text-lg font-semibold">Tipos de cambio</h3>
          <DataTable headers={["Tipo", "Compra", "Venta", "Var.", "Fuente", "Estado"]} rows={fxRows} />
        </div>
        <div className="min-w-0 rounded-lg border border-obs-border bg-obs-card p-4">
          {chartEntries.length ? (
            <Suspense fallback={<DataState title="Preparando grafico" />}>
              <DarkChart
                title="Valores disponibles"
                labels={chartEntries.map((entry) => entry.label)}
                values={chartEntries.map((entry) => entry.value)}
              />
            </Suspense>
          ) : (
            <DataState title="Gráfico no disponible" detail="No hay datos reales suficientes para graficar." />
          )}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-obs-border bg-obs-card p-4">
          <div className="text-xs uppercase tracking-wide text-obs-muted">Bandas cambiarias</div>
          {bands ? (
            <div className="mt-3 space-y-2 text-sm">
              <div className="flex justify-between gap-3"><span>Inferior</span><strong>{moneyFormatter.format(bands.banda_inferior)}</strong></div>
              <div className="flex justify-between gap-3"><span>Superior</span><strong>{moneyFormatter.format(bands.banda_superior)}</strong></div>
              <div className="text-xs text-obs-muted">{bands.fuente} · {bands.fecha}</div>
            </div>
          ) : <DataState title="No disponible" />}
        </div>
        <div className="rounded-lg border border-obs-border bg-obs-card p-4">
          <div className="text-xs uppercase tracking-wide text-obs-muted">Brecha vs. oficial</div>
          {gaps ? (
            <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
              {Object.entries(gaps).map(([key, value]) => (
                <div key={key}>
                  <div className="text-obs-muted">{key.replace("_pct", "").toUpperCase()}</div>
                  <strong>{percent(value)}</strong>
                </div>
              ))}
            </div>
          ) : <DataState title="No disponible" />}
        </div>
        <div className="rounded-lg border border-obs-border bg-obs-card p-4">
          <div className="text-xs uppercase tracking-wide text-obs-muted">Lecturas automáticas</div>
          {readings.length ? (
            <div className="mt-3 space-y-3">
              {readings.slice(0, 3).map((reading: any) => (
                <div key={reading.dedup_key} className="border-l-2 border-obs-violet pl-3">
                  <div className="flex items-center justify-between gap-2 text-sm">
                    <span className="font-medium">{reading.title}</span>
                    <span className="text-xs text-obs-violet">{reading.score}/100</span>
                  </div>
                  <div className="mt-1 text-xs text-obs-muted">{reading.description}</div>
                </div>
              ))}
            </div>
          ) : <DataState title="Sin lecturas activas" />}
        </div>
      </section>
      <section className="grid gap-4 md:grid-cols-2">
        <button type="button" className="rounded-lg border border-obs-border bg-obs-card p-5 text-left hover:border-obs-blue" onClick={() => onNavigate?.("news")}>
          <div className="font-medium">Noticias económicas</div>
          <div className="mt-1 text-sm text-obs-muted">Fuentes RSS directas</div>
        </button>
        <DataState title="Calendario económico" detail="No disponible" />
      </section>
      {payload.warnings.length ? <DataState title="Advertencias" detail={payload.warnings.join(" · ")} /> : null}
    </div>
  );
}
