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
  delivery: { status: string; channel: string };
  trigger_count: number;
  last_value: number | null;
  history: Array<{ id: number; value: number; reason: string; created_at: string }>;
}

export interface LegacyAlertItem {
  id: string;
  position: number;
  kind: string;
  title: string;
  asset: string | null;
  operator: string | null;
  target_value: number | null;
  unit: string | null;
  mode: string | null;
  segment: string | null;
  status: string;
  pause_scope: string | null;
  pause_until: string | null;
  armed: boolean | null;
  cooldown_seconds: number | null;
  tolerance_pct: number | null;
  last_triggered_at: string | null;
  last_trigger_value: number | null;
  logic: string | null;
  conditions: string[];
  source: string;
  write_surface: string;
}

export interface AlertCenterData {
  summary: {
    legacy_count: number;
    active_count: number;
    paused_count: number;
    draft_count: number;
    subscription_active: boolean;
  };
  legacy: {
    status: ApiStatus;
    source: string;
    read_only: boolean;
    items: LegacyAlertItem[];
    global_pause: boolean;
    global_pause_until: string | null;
    manage_command: string;
  };
  drafts: {
    status: ApiStatus;
    source: string;
    items: AlertItem[];
  };
  subscription: {
    status: ApiStatus;
    source: string;
    enabled: boolean;
    daily_time: string | null;
    timezone: string;
    channel: string;
    manage_command: string;
  };
  items: AlertItem[];
}

export interface SignalItem {
  id: number;
  type: string;
  title: string;
  description: string;
  severity: string;
  score: number;
  asset?: string | null;
  metric?: string | null;
  current_value?: number | null;
  previous_value?: number | null;
  variation?: number | null;
  source: string;
  status: string;
  timestamp: string;
}

export interface NewsItem {
  title: string;
  url: string;
  source: string;
  published_at: string | null;
  image_url: string | null;
}

export interface NewsData {
  items: NewsItem[];
  sources: string[];
}

export interface SummaryData {
  market: Record<string, any>;
  news: NewsData;
}

export interface SessionData {
  authentication: {
    mode: "telegram" | "development";
    validated: boolean;
    validation_method: string;
  };
  user: {
    id: string;
    username: string | null;
    first_name: string | null;
  };
  portfolio: {
    status: ApiStatus;
    source: string;
    item_count: number;
  };
  alerts?: {
    status: ApiStatus;
    source: string;
    rule_count: number;
  };
  subscription?: {
    status: ApiStatus;
    source: string;
    daily_time: string | null;
    enabled: boolean;
  };
  telegram_runtime: {
    enabled: boolean;
    state: string;
    token_configured: boolean;
    webhook_configured: boolean;
    keepalive_enabled: boolean;
    error: string | null;
  };
}

export interface PortfolioMoney {
  value: number;
  currency: string;
}

export interface PortfolioPosition {
  symbol: string;
  type: string | null;
  currency: string;
  quantity: number;
  invested_value: number;
  current_value: number;
  result: number;
  result_pct: number | null;
  weight: number;
  price_native: number | null;
  price_base: number | null;
  daily_change_pct: number | null;
  volatility_pct: number | null;
  added_timestamp: number | null;
  valuation_mode: string;
  projection: Record<string, number | null>;
}

export interface PortfolioSummary {
  configuration: Block<{
    base_currency: string;
    fx_reference: string | null;
    fx_value: number | null;
    fx_timestamp: number | null;
    benchmark: string | null;
  }>;
  budget: Block<PortfolioMoney>;
  invested_value: Block<PortfolioMoney>;
  current_value: Block<PortfolioMoney & {
    coverage_pct: number;
    valued_items: number;
    total_items: number;
  }>;
  result_absolute: Block<PortfolioMoney>;
  total_return: Block<{ value_pct: number; basis: string; invested_value: number }>;
  twr: Block<{ value_pct: number; method: string }>;
  mwr: Block<{ value_pct: number; method: string; cash_flow_count: number }>;
  benchmark: Block<{
    label: string;
    symbol: string;
    last_value: number | null;
    daily_change_pct: number | null;
  }>;
  drawdown: Block<{ value_pct: number }>;
  volatility: Block<{ value_pct: number; method: string }>;
  composition: Block<PortfolioPosition[]>;
  currency_exposure: Block<Array<{ currency: string; value: number; weight: number }>>;
  contributors: Block<PortfolioPosition[]>;
  detractors: Block<PortfolioPosition[]>;
  projection: Block<{ "3m": number | null; "6m": number | null; method: string }>;
  rebalance: Block<unknown>;
  valuation: Block<{
    as_of: number | null;
    coverage_pct: number;
    valued_items: number;
    total_items: number;
    method: string;
  }>;
}
