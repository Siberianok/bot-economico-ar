import { API_BASE_URL, API_CONFIG_ERROR } from "../api/client";
import { getConfig } from "../api/config";
import { getSession } from "../api/session";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { DataState } from "../components/States/DataState";
import { useEnvelope } from "../hooks/useEnvelope";
import { getTelegramStatus } from "../lib/telegram";

export function Settings() {
  const { payload, loading, error } = useEnvelope(getConfig);
  const { payload: sessionPayload, loading: sessionLoading } = useEnvelope(getSession);
  const telegram = getTelegramStatus();
  const serverSession = sessionPayload?.data;
  const telegramVerified = Boolean(serverSession?.authentication.validated);
  const alertsStatus = telegramVerified
    ? serverSession?.alerts?.status || "not_available"
    : "not_available";
  const subscriptionStatus = telegramVerified
    ? serverSession?.subscription?.enabled
      ? "active"
      : serverSession?.subscription?.status || "not_available"
    : "not_available";

  return (
    <div className="space-y-5">
      <header>
        <div className="text-xs uppercase tracking-wide text-obs-blue">Settings</div>
        <h2 className="text-3xl font-semibold">Entorno y preferencias</h2>
      </header>
      {loading ? <DataState title="Cargando configuración" /> : null}
      {error ? <DataState title="Error de carga" detail={error} /> : null}
      <section className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-obs-border bg-obs-card p-4">
          <h3 className="mb-3 text-lg font-semibold">API</h3>
          <div className="space-y-2 text-sm text-obs-muted">
            <div>Base URL: {API_BASE_URL || "No configurada"}</div>
            {API_CONFIG_ERROR ? <div>Configuración: {API_CONFIG_ERROR}</div> : null}
            <div className="flex gap-2">Estado: {payload ? <StatusBadge status={payload.status} /> : "No cargado"}</div>
            <div>Fuente: {payload?.source || "-"}</div>
          </div>
        </div>
        <div className="rounded-lg border border-obs-border bg-obs-card p-4">
          <h3 className="mb-3 text-lg font-semibold">Telegram Mini App</h3>
          <div className="space-y-2 text-sm text-obs-muted">
            <div>SDK cargado: {telegram.sdkLoaded ? "sí" : "no"}</div>
            <div>Contexto Telegram: {telegram.detected ? "detectado" : "modo desarrollo"}</div>
            <div>initData: {telegram.initDataAvailable ? "disponible" : "no disponible"}</div>
            <div className="flex items-center gap-2">
              Validación backend:
              {sessionLoading ? (
                "Comprobando"
              ) : (
                <StatusBadge status={serverSession?.authentication.validated ? "validated" : "development"} />
              )}
            </div>
            <div>
              Usuario: {serverSession?.user.first_name || serverSession?.user.username || telegram.userId}
            </div>
            <div className="flex items-center gap-2">
              Runtime Telegram:
              <StatusBadge status={serverSession?.telegram_runtime.state || "unknown"} />
            </div>
            <div className="flex items-center gap-2">
              Portfolio vinculado:
              <StatusBadge status={serverSession?.portfolio.status || "not_available"} />
            </div>
            <div className="flex items-center gap-2">
              Alertas Telegram:
              <StatusBadge status={alertsStatus} />
              <span>{telegramVerified ? serverSession?.alerts?.rule_count || 0 : "Sin vincular"}</span>
            </div>
            <div className="flex items-center gap-2">
              Resumen diario:
              <StatusBadge status={subscriptionStatus} />
              <span>{telegramVerified ? serverSession?.subscription?.daily_time || "Sin horario" : "Sin vincular"}</span>
            </div>
          </div>
        </div>
      </section>
      <DataState
        title="Preferencias"
        detail="Moneda base, benchmark y umbral de señales están disponibles vía API; edición visual avanzada queda para la siguiente fase."
      />
    </div>
  );
}
