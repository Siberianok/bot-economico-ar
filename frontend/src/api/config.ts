import { apiGet, apiPatch } from "./client";

export function getConfig() {
  return apiGet<Record<string, unknown>>("/api/v1/config");
}

export function updateConfig(body: Record<string, unknown>) {
  return apiPatch<Record<string, unknown>>("/api/v1/config", body);
}

