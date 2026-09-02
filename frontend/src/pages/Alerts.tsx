import {
  BellRing,
  Clock3,
  Database,
  History,
  LockKeyhole,
  Pause,
  Play,
  Radio,
  Save,
  ShieldCheck,
  Trash2,
} from "lucide-react";
import { FormEvent, useState } from "react";
import { createAlert, deleteAlert, getAlerts, updateAlert } from "../api/alerts";
import { getSession } from "../api/session";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { SegmentedTabs } from "../components/Buttons/SegmentedTabs";
import { DataState } from "../components/States/DataState";
import { useEnvelope } from "../hooks/useEnvelope";
import type { LegacyAlertItem } from "../types/api";

const kindLabels: Record<string, string> = {
  fx: "Dólar",
  metric: "Macro",
  ticker: "Mercado",
  crypto: "Cripto",
  composite: "Compuesta",
};

function formatValue(value: number | null, unit: string | null) {
  if (value == null) return "Sin umbral único";
  const formatted = new Intl.NumberFormat("es-AR", { maximumFractionDigits: 4 }).format(value);
  return unit === "ARS" ? `$ ${formatted}` : `${formatted}${unit ? ` ${unit}` : ""}`;
}

function formatDate(value: string | null) {
  if (!value) return "Sin registros";
  return new Intl.DateTimeFormat("es-AR", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(new Date(value));
}

function formatCooldown(seconds: number | null) {
  if (seconds == null) return "No informado";
  if (seconds >= 3600 && seconds % 3600 === 0) return `${seconds / 3600} h`;
  return `${Math.round(seconds / 60)} min`;
}

function LegacyRuleCard({ item }: { item: LegacyAlertItem }) {
  return (
    <article className="rounded-lg border border-obs-border bg-obs-card p-4">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs uppercase text-obs-blue">{kindLabels[item.kind] || item.kind}</span>
            <StatusBadge status={item.status} />
            <StatusBadge status="telegram" />
          </div>
          <h3 className="mt-2 text-lg font-semibold">{item.title}</h3>
          <div className="mt-1 text-sm text-obs-muted">
            {item.operator ? `${item.operator} ` : ""}{formatValue(item.target_value, item.unit)}
          </div>
          {item.conditions.length ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {item.conditions.map((condition) => (
                <span key={condition} className="rounded border border-obs-border px-2 py-1 text-xs text-obs-muted">
                  {condition}
                </span>
              ))}
            </div>
          ) : null}
        </div>
        <div className="grid min-w-64 grid-cols-2 gap-x-6 gap-y-3 text-sm">
          <div><div className="text-xs text-obs-muted">Cooldown</div><div>{formatCooldown(item.cooldown_seconds)}</div></div>
          <div><div className="text-xs text-obs-muted">Tolerancia</div><div>{item.tolerance_pct == null ? "No informada" : `${item.tolerance_pct}%`}</div></div>
          <div><div className="text-xs text-obs-muted">Estado interno</div><div>{item.armed == null ? "No aplica" : item.armed ? "Armada" : "Rearme pendiente"}</div></div>
          <div><div className="text-xs text-obs-muted">Último disparo</div><div>{formatDate(item.last_triggered_at)}</div></div>
        </div>
      </div>
      {item.pause_until ? <div className="mt-3 text-xs text-obs-warning">Pausa hasta {formatDate(item.pause_until)}</div> : null}
    </article>
  );
}

export function Alerts() {
  const [refresh, setRefresh] = useState(0);
  const [tab, setTab] = useState("telegram");
  const [category, setCategory] = useState("market");
  const [asset, setAsset] = useState("");
  const [metric, setMetric] = useState("price");
  const [target, setTarget] = useState("");
  const [operator, setOperator] = useState(">=");
  const [condition, setCondition] = useState<"exact" | "approximation">("exact");
  const [tolerance, setTolerance] = useState("0.5");
  const [cooldown, setCooldown] = useState("15");
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<number | null>(null);
  const { payload, loading, error } = useEnvelope(getAlerts, [refresh]);
  const { payload: sessionPayload, loading: sessionLoading } = useEnvelope(getSession);

  const center = payload?.data;
  const telegramVerified = Boolean(sessionPayload?.data?.authentication.validated);
  const legacyItems = telegramVerified ? center?.legacy?.items || [] : [];
  const subscriptionStatus = telegramVerified
    ? center?.subscription?.enabled
      ? "active"
      : center?.subscription?.status || "not_available"
    : "not_available";
  const draftItems = center?.drafts?.items || center?.items || [];
  const draftHistory = draftItems.flatMap((item) =>
    item.history.map((event) => ({
      id: `draft-${item.id}-${event.id}`,
      asset: item.asset,
      reason: event.reason,
      value: event.value,
      timestamp: event.created_at,
      source: "Borrador SQLite",
    })),
  );
  const legacyHistory = legacyItems
    .filter((item) => item.last_triggered_at)
    .map((item) => ({
      id: `${item.id}-${item.last_triggered_at}`,
      asset: item.title,
      reason: "Último disparo conocido",
      value: item.last_trigger_value,
      timestamp: item.last_triggered_at as string,
      source: "Telegram legacy",
    }));
  const historyItems = [...legacyHistory, ...draftHistory].sort(
    (left, right) => new Date(right.timestamp).getTime() - new Date(left.timestamp).getTime(),
  );

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setSaveMessage(null);
    setSaveError(null);
    try {
      const result = await createAlert({
        category,
        asset: asset.trim(),
        metric: metric.trim(),
        target_value: Number(target),
        condition_type: condition,
        operator,
        prealert_tolerance_pct: condition === "approximation" ? Number(tolerance) : null,
        cooldown_minutes: Number(cooldown),
      });
      if (result.status === "error") {
        setSaveError(result.warnings.join(" · ") || "No se pudo guardar el borrador.");
        return;
      }
      setTarget("");
      setSaveMessage("Regla guardada como borrador local; todavía no envía mensajes.");
      setTab("drafts");
      setRefresh((value) => value + 1);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "No se pudo guardar el borrador.");
    } finally {
      setSaving(false);
    }
  }

  async function toggleStatus(alertId: number, status: string) {
    setSaveError(null);
    const result = await updateAlert(alertId, { status: status === "paused" ? "draft" : "paused" });
    if (result.status === "error") setSaveError(result.warnings.join(" · "));
    setRefresh((value) => value + 1);
  }

  async function remove(alertId: number) {
    if (pendingDelete !== alertId) {
      setPendingDelete(alertId);
      return;
    }
    const result = await deleteAlert(alertId);
    if (result.status === "error") setSaveError(result.warnings.join(" · "));
    setPendingDelete(null);
    setRefresh((value) => value + 1);
  }

  return (
    <div className="space-y-5">
      <header className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="text-xs uppercase text-obs-blue">Alertas</div>
          <h2 className="text-3xl font-semibold">Centro de control</h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={payload?.status || "not_available"} />
          <span className="text-sm text-obs-muted">
            {center?.summary?.active_count || 0} activas · {center?.summary?.paused_count || 0} pausadas · {center?.summary?.draft_count ?? draftItems.length} borradores
          </span>
        </div>
      </header>

      {telegramVerified && center?.legacy?.status === "ok" ? (
        <div className="flex items-start gap-3 border-l-2 border-obs-positive bg-obs-card2 p-4">
          <ShieldCheck className="mt-0.5 shrink-0 text-obs-positive" size={20} />
          <div><div className="font-medium">Estado Telegram vinculado</div><div className="text-sm text-obs-muted">Las reglas operativas se muestran sin modificar su estado.</div></div>
        </div>
      ) : (
        <div className="flex items-start gap-3 border-l-2 border-obs-warning bg-obs-card2 p-4">
          <LockKeyhole className="mt-0.5 shrink-0 text-obs-warning" size={20} />
          <div>
            <div className="font-medium">{sessionLoading ? "Verificando identidad Telegram" : "Estado Telegram no vinculado"}</div>
            <div className="text-sm text-obs-muted">{sessionLoading ? "Comprobando la sesión con el backend." : "Abrí la Mini App desde /dashboard para consultar ALERTS y SUBS de tu cuenta."}</div>
          </div>
        </div>
      )}

      <SegmentedTabs
        label="Secciones de alertas"
        value={tab}
        onChange={setTab}
        items={[
          { key: "telegram", label: "Telegram" },
          { key: "drafts", label: "Borradores" },
          { key: "new", label: "Nueva" },
          { key: "subscription", label: "Suscripción" },
          { key: "history", label: "Historial" },
        ]}
      />

      {loading ? <DataState title="Cargando alertas" /> : null}
      {error ? <DataState title="Error de carga" detail={error} /> : null}
      {saveMessage ? <DataState title="Borrador actualizado" detail={saveMessage} /> : null}
      {saveError ? <DataState title="No se pudo completar la acción" detail={saveError} /> : null}

      {tab === "telegram" ? (
        legacyItems.length ? (
          <div className="grid gap-3">{legacyItems.map((item) => <LegacyRuleCard key={item.id} item={item} />)}</div>
        ) : (
          <DataState
            title={telegramVerified && center?.legacy?.status === "ok" ? "Sin alertas Telegram" : "Alertas Telegram no vinculadas"}
            detail={telegramVerified && center?.legacy?.status === "ok" ? "La cuenta vinculada no tiene reglas operativas guardadas." : "Se requiere una identidad Telegram firmada para leerlas."}
          />
        )
      ) : null}

      {tab === "drafts" ? (
        draftItems.length ? (
          <div className="grid gap-3">
            {draftItems.map((item) => (
              <article key={item.id} className="rounded-lg border border-obs-border bg-obs-card p-4">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2"><Database size={16} className="text-obs-violet" /><h3 className="font-semibold">{item.asset}</h3><StatusBadge status={item.status} /><StatusBadge status={item.delivery.status} /></div>
                    <div className="mt-2 text-sm text-obs-muted">{item.metric} · {item.condition_type} · {item.operator} {item.target_value} · cooldown {item.cooldown_minutes} min</div>
                    <div className="mt-1 text-xs text-obs-muted">{item.trigger_count} comprobaciones disparadas</div>
                  </div>
                  <div className="flex gap-2">
                    <button type="button" title={item.status === "paused" ? "Reanudar borrador" : "Pausar borrador"} className="flex size-10 items-center justify-center rounded-md border border-obs-border text-obs-muted hover:text-obs-blue" onClick={() => toggleStatus(item.id, item.status)}>
                      {item.status === "paused" ? <Play size={17} /> : <Pause size={17} />}
                    </button>
                    <button type="button" title={pendingDelete === item.id ? "Confirmar eliminación" : "Eliminar borrador"} className={`flex min-h-10 items-center justify-center rounded-md border px-3 ${pendingDelete === item.id ? "border-obs-negative text-obs-negative" : "border-obs-border text-obs-muted hover:text-obs-negative"}`} onClick={() => remove(item.id)}>
                      <Trash2 size={17} />{pendingDelete === item.id ? <span className="ml-2 text-xs">Confirmar</span> : null}
                    </button>
                  </div>
                </div>
              </article>
            ))}
          </div>
        ) : <DataState title="Sin borradores" detail="La base SQLite no contiene reglas locales para esta identidad." />
      ) : null}

      {tab === "new" ? (
        <form className="grid gap-4 rounded-lg border border-obs-border bg-obs-card p-4 md:grid-cols-2 xl:grid-cols-4" onSubmit={submit}>
          <div className="flex items-start gap-3 border-b border-obs-border pb-4 md:col-span-2 xl:col-span-4"><LockKeyhole size={18} className="mt-0.5 text-obs-violet" /><div><div className="font-medium">Borrador local</div><div className="text-sm text-obs-muted">Se guarda en SQLite y permanece separado de las reglas Telegram operativas.</div></div></div>
          <label className="grid gap-1 text-xs text-obs-muted">Categoría<select className="field" value={category} onChange={(event) => setCategory(event.target.value)}><option value="market">Mercado</option><option value="equity">Acción</option><option value="cedear">CEDEAR</option><option value="crypto">Cripto</option><option value="macro">Macro</option></select></label>
          <label className="grid gap-1 text-xs text-obs-muted">Activo<input className="field" value={asset} onChange={(event) => setAsset(event.target.value)} placeholder="GGAL, CCL, BTC" required /></label>
          <label className="grid gap-1 text-xs text-obs-muted">Métrica<input className="field" value={metric} onChange={(event) => setMetric(event.target.value)} placeholder="price" required /></label>
          <label className="grid gap-1 text-xs text-obs-muted">Condición<div className="grid grid-cols-[0.7fr_1.3fr] gap-2"><select className="field" value={operator} onChange={(event) => setOperator(event.target.value)} aria-label="Operador"><option>&gt;=</option><option>&lt;=</option><option>&gt;</option><option>&lt;</option><option>==</option></select><input className="field" value={target} onChange={(event) => setTarget(event.target.value)} placeholder="Objetivo" type="number" step="any" required /></div></label>
          <label className="grid gap-1 text-xs text-obs-muted">Tipo<select className="field" value={condition} onChange={(event) => setCondition(event.target.value as "exact" | "approximation")}><option value="exact">Exacta</option><option value="approximation">Aproximación</option></select></label>
          <label className="grid gap-1 text-xs text-obs-muted">Tolerancia %<input className="field" value={tolerance} onChange={(event) => setTolerance(event.target.value)} type="number" min="0" max="100" step="0.1" disabled={condition === "exact"} /></label>
          <label className="grid gap-1 text-xs text-obs-muted">Cooldown<select className="field" value={cooldown} onChange={(event) => setCooldown(event.target.value)}><option value="15">15 minutos</option><option value="60">1 hora</option><option value="360">6 horas</option><option value="1440">24 horas</option></select></label>
          <div className="flex items-end"><button className="flex min-h-10 w-full items-center justify-center gap-2 rounded-md bg-obs-blue px-4 text-sm font-semibold text-obs-bg disabled:opacity-60" type="submit" disabled={saving}><Save size={17} /> {saving ? "Guardando" : "Guardar borrador"}</button></div>
        </form>
      ) : null}

      {tab === "subscription" ? (
        <section className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-lg border border-obs-border bg-obs-card p-5">
            <div className="flex items-center justify-between gap-3"><div className="flex items-center gap-2"><Clock3 className="text-obs-blue" size={20} /><h3 className="font-semibold">Resumen diario</h3></div><StatusBadge status={subscriptionStatus} /></div>
            <div className="mt-6 text-4xl font-semibold">{telegramVerified ? center?.subscription?.daily_time || "--:--" : "--:--"}</div>
            <div className="mt-2 text-sm text-obs-muted">{telegramVerified ? center?.subscription?.timezone || "Hora Argentina" : "Vinculación Telegram requerida"}</div>
          </div>
          <div className="rounded-lg border border-obs-border bg-obs-card p-5">
            <div className="flex items-center gap-2"><Radio className="text-obs-violet" size={20} /><h3 className="font-semibold">Entrega</h3></div>
            <div className="mt-5 text-sm"><div className="text-obs-muted">Canal</div><div className="mt-1">Telegram</div></div>
            <div className="mt-4 text-sm"><div className="text-obs-muted">Gestión</div><div className="mt-1 font-medium text-obs-blue">/subs</div></div>
          </div>
        </section>
      ) : null}

      {tab === "history" ? (
        historyItems.length ? (
          <div className="divide-y divide-obs-border border-y border-obs-border">
            {historyItems.map((event) => (
              <div key={event.id} className="flex min-h-20 items-center gap-3 py-3 text-sm">
                <History size={17} className="shrink-0 text-obs-muted" />
                <div className="min-w-0 flex-1"><div className="font-medium">{event.asset}</div><div className="text-xs text-obs-muted">{event.reason} · {formatDate(event.timestamp)} · {event.source}</div></div>
                <div>{event.value == null ? "--" : event.value}</div>
              </div>
            ))}
          </div>
        ) : <DataState title="Sin disparos registrados" detail="El estado legacy conserva únicamente el último disparo conocido por regla." />
      ) : null}

      {payload?.warnings.length ? (
        <div className="rounded-lg border border-obs-border bg-obs-card2 p-4 text-sm"><div className="flex items-center gap-2 font-medium"><BellRing size={17} className="text-obs-warning" />Estado del sistema</div><div className="mt-2 text-obs-muted">{payload.warnings.join(" · ")}</div></div>
      ) : null}
    </div>
  );
}
