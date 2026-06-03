import { API_BASE_URL, API_CONFIG_ERROR } from "../api/client";
import { getConfig } from "../api/config";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { DataState } from "../components/States/DataState";
import { useEnvelope } from "../hooks/useEnvelope";
import { getTelegramStatus } from "../lib/telegram";

export function Settings() {
  const { payload, loading, error } = useEnvelope(getConfig);
  const telegram = getTelegramStatus();

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
            <div>Detectada: {telegram.detected ? "sí" : "no"}</div>
            <div>initData: {telegram.initDataAvailable ? "disponible" : "no disponible"}</div>
            <div>Usuario/dev: {telegram.userId}</div>
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
