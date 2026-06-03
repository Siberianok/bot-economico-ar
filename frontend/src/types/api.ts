export type ApiStatus = "ok" | "partial" | "not_available" | "error";
export type Freshness = "current" | "stale" | "fallback" | "unknown";

export interface ApiEnvelope<T> {
  status: ApiStatus;
  timestamp: string;
  source: string;
  freshness: Freshness;
  data: T;
  warnings: string[];
}

export interface Block<T = unknown> {
  status: ApiStatus;
  data: T | null;
}

export interface ScreenerItem {
  symbol: string;
  status: ApiStatus;
  price: number | null;
  returns: Record<string, number | null>;
  score: number | null;
  momentum: number | null;
  risk: string | null;
  liquidity: number | null;
  projection: Record<string, number | null>;
  warnings?: string[];
}

export interface AlertItem {
  id: number;
  asset: string;
  metric: string;
  condition_type: string;
  operator: string;
  target_value: number;
  prealert_tolerance_pct: number | null;
  cooldown_minutes: number;
  channel: string;
  status: string;
  trigger_count: number;
  last_value: number | null;
  history: Array<{ id: number; value: number; reason: string; created_at: string }>;
}

export interface SignalItem {
  id: number;
  type: string;
  title: string;
  description: string;
  severity: string;
  score: number;
  source: string;
  status: string;
  timestamp: string;
}

