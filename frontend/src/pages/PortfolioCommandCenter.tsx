import {
  BarChart3,
  Download,
  Pencil,
  Plus,
  Scale,
  ShieldCheck,
  ShieldOff,
  SlidersHorizontal,
} from "lucide-react";
import { useState } from "react";
import { getPortfolioSummary } from "../api/portfolio";
import { getSession } from "../api/session";
import { StatusBadge, statusLabel } from "../components/Badges/StatusBadge";
import { SegmentedTabs } from "../components/Buttons/SegmentedTabs";
import { KpiCard } from "../components/Cards/KpiCard";
import {
  PortfolioDonutChart,
  PortfolioResultChart,
  type PortfolioChartItem,
} from "../components/Charts/PortfolioCharts";
import { DataState } from "../components/States/DataState";
import { useEnvelope } from "../hooks/useEnvelope";
import type { Block, PortfolioMoney, PortfolioPosition } from "../types/api";

const number = new Intl.NumberFormat("es-AR", { maximumFractionDigits: 2 });

function moneyValue(value: unknown, currency = "ARS"): string {
  if (typeof value !== "number") return "No disponible";
  return new Intl.NumberFormat("es-AR", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(value);
}

function moneyBlock(block?: Block<PortfolioMoney>): string {
  return moneyValue(block?.data?.value, block?.data?.currency || "ARS");
}

function percent(value: unknown): string {
  return typeof value === "number" ? `${number.format(value)}%` : "No disponible";
}

function tone(value: unknown): "positive" | "negative" | "neutral" {
  if (typeof value !== "number") return "neutral";
  return value >= 0 ? "positive" : "negative";
}

export function PortfolioCommandCenter() {
  const [tab, setTab] = useState("summary");
  const { payload, loading, error } = useEnvelope(getPortfolioSummary);
  const { payload: sessionPayload } = useEnvelope(getSession);
  const data = payload?.data;
  const session = sessionPayload?.data;
  const configuration = data?.configuration.data;
  const baseCurrency = configuration?.base_currency || "ARS";
  const composition = data?.composition.data || [];
  const exposure = data?.currency_exposure.data || [];
  const totalReturn = data?.total_return.data?.value_pct;
  const resultAbsolute = data?.result_absolute.data?.value;
  const telegramVerified = Boolean(session?.authentication.validated);
  const resultRows: PortfolioChartItem[] = [];
  const seenResultSymbols = new Set<string>();
  for (const item of [
    ...(data?.contributors.data || []).slice(0, 3),
    ...(data?.detractors.data || []).slice(0, 3),
  ]) {
    if (!seenResultSymbols.has(item.symbol)) {
      seenResultSymbols.add(item.symbol);
      resultRows.push({ label: item.symbol, value: item.result });
    }
  }

  const toolRows = [
    { label: "Base y moneda", icon: SlidersHorizontal, status: data?.configuration.status },
    { label: "Presupuesto", icon: Scale, status: data?.budget.status },
    { label: "Agregar instrumento", icon: Plus, status: "telegram" },
    {
      label: "Editar instrumento",
      icon: Pencil,
      status: composition.length ? "telegram" : "not_available",
    },
    { label: "Rebalanceo y simulación", icon: BarChart3, status: data?.rebalance.status },
    { label: "Exportar", icon: Download, status: "telegram" },
  ];

  return (
    <div className="space-y-5">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-xs uppercase tracking-wide text-obs-blue">Portfolio</div>
          <h2 className="text-3xl font-semibold">Command Center</h2>
        </div>
        {payload ? (
          <div className="flex gap-2">
            <StatusBadge status={payload.status} />
            <StatusBadge status={payload.freshness} />
          </div>
        ) : null}
      </header>

      <section
        className={`flex items-start gap-3 border-l-2 px-4 py-3 ${
          telegramVerified ? "border-obs-positive bg-obs-card" : "border-obs-warning bg-obs-card2"
        }`}
      >
        {telegramVerified ? (
          <ShieldCheck className="mt-0.5 shrink-0 text-obs-positive" size={20} />
        ) : (
          <ShieldOff className="mt-0.5 shrink-0 text-obs-warning" size={20} />
        )}
        <div className="min-w-0">
          <div className="font-medium">
            {telegramVerified ? "Identidad Telegram verificada" : "Sesión local de desarrollo"}
          </div>
          <div className="mt-1 text-sm text-obs-muted">
            {telegramVerified
              ? `${session?.user.first_name || session?.user.username || "Usuario"} · ${session?.portfolio.item_count || 0} posiciones legacy`
              : "El portfolio real se vincula al abrir esta pantalla desde /dashboard en Telegram."}
          </div>
        </div>
      </section>

      {loading ? <DataState title="Cargando portfolio" /> : null}
      {error ? <DataState title="Error de carga" detail={error} /> : null}

      <SegmentedTabs
        label="Secciones del portfolio"
        value={tab}
        onChange={setTab}
        items={[
          { key: "summary", label: "Resumen" },
          { key: "positions", label: "Posiciones" },
          { key: "analytics", label: "Analítica" },
          { key: "management", label: "Gestión" },
        ]}
      />

      {payload?.status === "not_available" ? (
        <DataState
          title={telegramVerified ? "Portfolio todavía no configurado" : "Portfolio no vinculado"}
          detail={
            telegramVerified
              ? "La identidad es válida, pero no tiene un PF legacy cargado. Configuralo desde /portafolio."
              : "Esta sesión no puede consultar el PF de ningún usuario Telegram."
          }
        />
      ) : null}

      {tab === "summary" ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <KpiCard title="Valor invertido" value={moneyBlock(data?.invested_value)} />
            <KpiCard
              title="Valor actual"
              value={moneyBlock(data?.current_value)}
              meta={
                data?.current_value.data?.coverage_pct != null
                  ? `${number.format(data.current_value.data.coverage_pct)}% valuado`
                  : undefined
              }
            />
            <KpiCard
              title="Resultado"
              value={moneyValue(resultAbsolute, baseCurrency)}
              meta={percent(totalReturn)}
              tone={tone(resultAbsolute)}
            />
            <KpiCard
              title="Proyección 3M"
              value={percent(data?.projection.data?.["3m"])}
              tone="blue"
            />
          </div>

          {composition.length ? (
            <section className="grid gap-4 lg:grid-cols-2">
              <div className="min-w-0 rounded-lg border border-obs-border bg-obs-card p-4">
                <h3 className="text-lg font-semibold">Composición actual</h3>
                <PortfolioDonutChart
                  items={composition.map((item) => ({
                    label: item.symbol,
                    value: item.current_value,
                  }))}
                />
              </div>
              <div className="min-w-0 rounded-lg border border-obs-border bg-obs-card p-4">
                <h3 className="text-lg font-semibold">Exposición monetaria</h3>
                <PortfolioDonutChart
                  items={exposure.map((item) => ({ label: item.currency, value: item.value }))}
                />
              </div>
            </section>
          ) : null}
        </>
      ) : null}

      {tab === "positions" ? (
        <section className="space-y-3">
          {composition.length ? (
            composition.map((item: PortfolioPosition) => (
              <article
                key={item.symbol}
                className="rounded-lg border border-obs-border bg-obs-card p-4"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="font-semibold">{item.symbol}</h3>
                    <div className="mt-1 text-xs text-obs-muted">
                      {item.type || "Instrumento"} · {item.currency} · {number.format(item.quantity)} unidades
                    </div>
                  </div>
                  <div className={`text-right font-semibold ${item.result >= 0 ? "text-obs-positive" : "text-obs-negative"}`}>
                    {moneyValue(item.result, baseCurrency)}
                    <div className="text-xs">{percent(item.result_pct)}</div>
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                  <div><span className="block text-xs text-obs-muted">Invertido</span>{moneyValue(item.invested_value, baseCurrency)}</div>
                  <div><span className="block text-xs text-obs-muted">Actual</span>{moneyValue(item.current_value, baseCurrency)}</div>
                  <div><span className="block text-xs text-obs-muted">Precio</span>{moneyValue(item.price_native, item.currency)}</div>
                  <div><span className="block text-xs text-obs-muted">Peso</span>{percent(item.weight * 100)}</div>
                </div>
                <div className="mt-3 h-1.5 overflow-hidden rounded bg-obs-card2">
                  <div
                    className="h-full rounded bg-obs-blue"
                    style={{ width: `${Math.min(100, Math.max(0, item.weight * 100))}%` }}
                  />
                </div>
              </article>
            ))
          ) : (
            <DataState
              title="Posiciones no disponibles"
              detail={statusLabel(data?.composition.status || "not_available")}
            />
          )}
        </section>
      ) : null}

      {tab === "analytics" ? (
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <KpiCard
              title="TWR aproximado"
              value={percent(data?.twr.data?.value_pct)}
              tone={tone(data?.twr.data?.value_pct)}
              meta={statusLabel(data?.twr.status || "not_available")}
            />
            <KpiCard
              title="MWR aproximado"
              value={percent(data?.mwr.data?.value_pct)}
              tone={tone(data?.mwr.data?.value_pct)}
              meta={statusLabel(data?.mwr.status || "not_available")}
            />
            <KpiCard
              title="Volatilidad"
              value={percent(data?.volatility.data?.value_pct)}
              meta="Ponderada por posición"
            />
            <KpiCard
              title={data?.benchmark.data?.label || "Benchmark"}
              value={percent(data?.benchmark.data?.daily_change_pct)}
              meta="Variación diaria de referencia"
              tone={tone(data?.benchmark.data?.daily_change_pct)}
            />
          </div>
          <div className="grid gap-4 lg:grid-cols-[1.35fr_0.65fr]">
            <section className="min-w-0 rounded-lg border border-obs-border bg-obs-card p-4">
              <h3 className="text-lg font-semibold">Aportantes y detractores</h3>
              {resultRows.length ? (
                <PortfolioResultChart items={resultRows} />
              ) : (
                <DataState title="Resultados por instrumento no disponibles" />
              )}
            </section>
            <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
              <KpiCard title="Escenario 3M" value={percent(data?.projection.data?.["3m"])} tone="blue" />
              <KpiCard title="Escenario 6M" value={percent(data?.projection.data?.["6m"])} tone="blue" />
              <DataState
                title="Drawdown"
                detail={statusLabel(data?.drawdown.status || "not_available")}
              />
            </section>
          </div>
          <DataState
            title="Metodología"
            detail="TWR y MWR se muestran como aproximaciones parciales de la foto actual. El cálculo canónico requiere historial de valuaciones."
          />
        </div>
      ) : null}

      {tab === "management" ? (
        <section className="grid gap-5 lg:grid-cols-[0.8fr_1.2fr]">
          <div className="rounded-lg border border-obs-border bg-obs-card p-4">
            <h3 className="text-lg font-semibold">Configuración</h3>
            <dl className="mt-4 space-y-3 text-sm">
              <div className="flex justify-between gap-3"><dt className="text-obs-muted">Moneda base</dt><dd>{baseCurrency}</dd></div>
              <div className="flex justify-between gap-3"><dt className="text-obs-muted">Tipo de cambio</dt><dd>{configuration?.fx_reference || "-"}</dd></div>
              <div className="flex justify-between gap-3"><dt className="text-obs-muted">Benchmark</dt><dd>{configuration?.benchmark || "-"}</dd></div>
              <div className="flex justify-between gap-3"><dt className="text-obs-muted">Presupuesto</dt><dd>{moneyBlock(data?.budget)}</dd></div>
            </dl>
          </div>
          <div className="divide-y divide-obs-border border-y border-obs-border">
            {toolRows.map((tool) => {
              const Icon = tool.icon;
              return (
                <div key={tool.label} className="flex min-h-14 items-center gap-3">
                  <Icon size={18} className="text-obs-muted" />
                  <span className="flex-1 text-sm font-medium">{tool.label}</span>
                  <StatusBadge status={tool.status || "not_available"} />
                </div>
              );
            })}
          </div>
        </section>
      ) : null}

      {payload?.warnings.length ? (
        <DataState title="Advertencias" detail={payload.warnings.join(" · ")} />
      ) : null}
    </div>
  );
}
