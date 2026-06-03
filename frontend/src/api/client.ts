import { getTelegramInitData } from "../lib/telegram";
import type { ApiEnvelope } from "../types/api";

const configuredApiBaseUrl = (import.meta.env.VITE_API_BASE_URL || "").trim();
const API_BASE_URL = configuredApiBaseUrl || (import.meta.env.PROD ? "" : "http://127.0.0.1:8000");
const API_CONFIG_ERROR =
  import.meta.env.PROD && !configuredApiBaseUrl
    ? "VITE_API_BASE_URL no configurado. Definí la URL HTTPS del backend en Render."
    : null;

function apiUrl(path: string): string {
  if (API_CONFIG_ERROR) {
    throw new Error(API_CONFIG_ERROR);
  }
  return `${API_BASE_URL}${path}`;
}

export async function apiGet<T>(path: string): Promise<ApiEnvelope<T>> {
  const initData = getTelegramInitData();
  const response = await fetch(apiUrl(path), {
    headers: {
      ...(initData ? { "X-Telegram-Init-Data": initData } : {}),
    },
  });
  const payload = (await response.json()) as ApiEnvelope<T>;
  if (!response.ok) {
    return payload;
  }
  return payload;
}

export async function apiPost<T>(path: string, body: unknown): Promise<ApiEnvelope<T>> {
  const initData = getTelegramInitData();
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(initData ? { "X-Telegram-Init-Data": initData } : {}),
    },
    body: JSON.stringify(body),
  });
  return (await response.json()) as ApiEnvelope<T>;
}

export async function apiPatch<T>(path: string, body: unknown): Promise<ApiEnvelope<T>> {
  const initData = getTelegramInitData();
  const response = await fetch(apiUrl(path), {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      ...(initData ? { "X-Telegram-Init-Data": initData } : {}),
    },
    body: JSON.stringify(body),
  });
  return (await response.json()) as ApiEnvelope<T>;
}

export { API_BASE_URL, API_CONFIG_ERROR };
