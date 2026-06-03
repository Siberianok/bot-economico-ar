import type { ReactNode } from "react";

interface Props {
  title: string;
  value: ReactNode;
  meta?: string;
  tone?: "neutral" | "positive" | "negative" | "warning" | "blue";
}

const tones = {
  neutral: "text-obs-text",
  positive: "text-obs-positive",
  negative: "text-obs-negative",
  warning: "text-obs-warning",
  blue: "text-obs-blue",
};

export function KpiCard({ title, value, meta, tone = "neutral" }: Props) {
  return (
    <section className="rounded-lg border border-obs-border bg-obs-card p-4">
      <div className="text-xs uppercase tracking-wide text-obs-muted">{title}</div>
      <div className={`mt-3 text-2xl font-semibold ${tones[tone]}`}>{value}</div>
      {meta ? <div className="mt-2 text-xs text-obs-muted">{meta}</div> : null}
    </section>
  );
}

