import { useState } from "react";
import { getScreener } from "../api/screener";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { DataState } from "../components/States/DataState";
import { DataTable } from "../components/Tables/DataTable";
import { useEnvelope } from "../hooks/useEnvelope";

export function Screener() {
  const [kind, setKind] = useState<"acciones" | "cedears">("acciones");
  const { payload, loading, error } = useEnvelope(() => getScreener(kind), [kind]);

  const formatNumber = (value: number | null | undefined) =>
    value == null ? "-" : new Intl.NumberFormat("es-AR", { maximumFractionDigits: 2 }).format(value);
  const formatPercent = (value: number | null | undefined) =>
    value == null ? "-" : `${formatNumber(value)}%`;
  const formatPrice = (value: number | null | undefined) =>
    value == null ? "No disponible" : `$ ${formatNumber(value)}`;

  const rows = payload?.data?.items?.map((item) => {
    const base = [item.symbol, formatPrice(item.price)];
    const returns = kind === "cedears"
      ? [
          formatPercent(item.returns?.["6m_ars"] ?? item.returns?.["6m"]),
          formatPercent(item.returns?.["6m_usd"]),
          formatPercent(item.returns?.["6m_fx"]),
        ]
      : [
          formatPercent(item.returns?.["1m"]),
          formatPercent(item.returns?.["3m"]),
          formatPercent(item.returns?.["6m"]),
        ];
    return [
      ...base,
      ...returns,
      formatNumber(item.score),
      item.risk ?? "No disponible",
      formatPercent(item.projection?.["3m"]),
      formatPercent(item.projection?.["6m"]),
      <StatusBadge status={item.status} />,
    ];
  }) || [];
  const returnHeaders = kind === "cedears" ? ["6M ARS", "6M USD", "Efecto FX"] : ["1M", "3M", "6M"];

  return (
    <div className="space-y-5">
      <header>
        <div className="text-xs uppercase tracking-wide text-obs-blue">Acciones & CEDEARs</div>
        <h2 className="text-3xl font-semibold">Screener</h2>
      </header>
      <div className="flex gap-2">
        {(["acciones", "cedears"] as const).map((item) => (
          <button
            key={item}
            className={`rounded-md border px-4 py-2 text-sm ${
              kind === item ? "border-obs-blue bg-obs-card2 text-obs-text" : "border-obs-border text-obs-muted"
            }`}
            onClick={() => setKind(item)}
          >
            {item === "acciones" ? "Acciones argentinas" : "CEDEARs"}
          </button>
        ))}
      </div>
      {loading ? <DataState title="Cargando screener" /> : null}
      {error ? <DataState title="Error de carga" detail={error} /> : null}
      {payload ? (
        <div className="min-w-0 rounded-lg border border-obs-border bg-obs-card p-4">
          <div className="mb-3 flex gap-2">
            <StatusBadge status={payload.status} />
            <StatusBadge status={payload.freshness} />
          </div>
          {rows.length ? (
            <DataTable
              headers={["Ticker", "Precio", ...returnHeaders, "Score", "Riesgo", "Proy. 3M", "Proy. 6M", "Estado"]}
              rows={rows}
            />
          ) : (
            <DataState title="Sin instrumentos disponibles" />
          )}
        </div>
      ) : null}
      {payload?.warnings.length ? <DataState title="Advertencias" detail={payload.warnings.join(" · ")} /> : null}
    </div>
  );
}
