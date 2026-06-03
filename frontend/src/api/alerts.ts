import { apiGet, apiPost } from "./client";
import type { AlertItem } from "../types/api";

export interface AlertsData {
  items: AlertItem[];
}

export function getAlerts() {
  return apiGet<AlertsData>("/api/v1/alerts");
}

export function createAlert(body: {
  asset: string;
  target_value: number;
  condition_type: "exact" | "approximation";
  operator: string;
  prealert_tolerance_pct?: number | null;
  cooldown_minutes: number;
}) {
  return apiPost<AlertItem>("/api/v1/alerts", body);
}

