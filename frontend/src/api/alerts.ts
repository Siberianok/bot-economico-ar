import { apiDelete, apiGet, apiPatch, apiPost } from "./client";
import type { AlertCenterData, AlertItem } from "../types/api";

export function getAlerts() {
  return apiGet<AlertCenterData>("/api/v1/alerts");
}

export function createAlert(body: {
  category?: string;
  asset: string;
  metric?: string;
  target_value: number;
  condition_type: "exact" | "approximation";
  operator: string;
  prealert_tolerance_pct?: number | null;
  cooldown_minutes: number;
}) {
  return apiPost<AlertItem>("/api/v1/alerts", body);
}

export function updateAlert(alertId: number, body: { status?: string; cooldown_minutes?: number }) {
  return apiPatch<AlertItem>(`/api/v1/alerts/${alertId}`, body);
}

export function deleteAlert(alertId: number) {
  return apiDelete<{ deleted: boolean; id: number }>(`/api/v1/alerts/${alertId}`);
}
