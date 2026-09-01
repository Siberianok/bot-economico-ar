import type { ApiStatus, Freshness } from "../../types/api";

interface Props {
  status: ApiStatus | Freshness | string;
}

const labels: Record<string, string> = {
  ok: "Disponible",
  current: "Actualizado",
  partial: "Parcial",
  fallback: "Respaldo",
  stale: "Desactualizado",
  error: "Error",
  not_available: "No disponible",
  unknown: "Sin verificar",
  draft: "Borrador",
  paused: "Pausada",
  not_connected: "Sin conexión",
  telegram: "Telegram",
  info: "Informativa",
  warning: "Atención",
  running: "Operativo",
  disabled: "Desactivado",
  development: "Desarrollo",
  validated: "Verificada",
  active: "Activa",
  waiting_rearm: "Esperando rearme",
  read_only: "Solo lectura",
};

export function statusLabel(status: string): string {
  return labels[status] || status;
}

export function StatusBadge({ status }: Props) {
  const style =
    status === "ok" || status === "current" || status === "running" || status === "validated" || status === "active"
      ? "border-obs-positive/50 bg-obs-positive/10 text-obs-positive"
      : status === "partial" || status === "fallback" || status === "waiting_rearm"
      ? "border-obs-warning/50 bg-obs-warning/10 text-obs-warning"
      : status === "error"
      ? "border-obs-negative/50 bg-obs-negative/10 text-obs-negative"
      : status === "draft" || status === "info"
      ? "border-obs-violet/50 bg-obs-violet/10 text-obs-violet"
      : "border-obs-border bg-obs-card2 text-obs-muted";
  return <span className={`rounded-full border px-2 py-1 text-xs ${style}`}>{statusLabel(status)}</span>;
}
