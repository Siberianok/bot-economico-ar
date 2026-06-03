import type { ApiStatus, Freshness } from "../../types/api";

interface Props {
  status: ApiStatus | Freshness | string;
}

export function StatusBadge({ status }: Props) {
  const style =
    status === "ok" || status === "current"
      ? "border-obs-positive/50 bg-obs-positive/10 text-obs-positive"
      : status === "partial" || status === "fallback"
      ? "border-obs-warning/50 bg-obs-warning/10 text-obs-warning"
      : status === "error"
      ? "border-obs-negative/50 bg-obs-negative/10 text-obs-negative"
      : "border-obs-border bg-obs-card2 text-obs-muted";
  return <span className={`rounded-full border px-2 py-1 text-xs ${style}`}>{status}</span>;
}

