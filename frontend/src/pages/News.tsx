import { ExternalLink, Newspaper } from "lucide-react";
import { getNews } from "../api/news";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { DataState } from "../components/States/DataState";
import { useEnvelope } from "../hooks/useEnvelope";

function published(value: string | null): string {
  if (!value) return "Fecha no informada";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("es-AR", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

export function News() {
  const { payload, loading, error } = useEnvelope(getNews);
  const items = payload?.data?.items || [];

  return (
    <div className="space-y-5">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-xs uppercase tracking-wide text-obs-blue">Noticias</div>
          <h2 className="text-3xl font-semibold">Economía y mercados</h2>
        </div>
        {payload ? (
          <div className="flex gap-2"><StatusBadge status={payload.status} /><StatusBadge status={payload.freshness} /></div>
        ) : null}
      </header>
      {loading ? <DataState title="Actualizando noticias" /> : null}
      {error ? <DataState title="Error de carga" detail={error} /> : null}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {items.map((item) => (
          <article key={item.url} className="overflow-hidden rounded-lg border border-obs-border bg-obs-card">
            {item.image_url ? (
              <img
                src={item.image_url}
                alt=""
                className="aspect-[16/8] w-full object-cover"
                loading="lazy"
                onError={(event) => { event.currentTarget.style.display = "none"; }}
              />
            ) : (
              <div className="flex aspect-[16/5] items-center justify-center bg-obs-card2 text-obs-muted"><Newspaper size={24} /></div>
            )}
            <div className="p-4">
              <div className="text-xs text-obs-muted">{item.source} · {published(item.published_at)}</div>
              <h3 className="mt-2 text-base font-semibold">{item.title}</h3>
              <a className="mt-4 inline-flex items-center gap-2 text-sm text-obs-blue" href={item.url} target="_blank" rel="noreferrer">
                Abrir nota <ExternalLink size={15} />
              </a>
            </div>
          </article>
        ))}
      </div>
      {!loading && !items.length ? <DataState title="Noticias no disponibles" detail={payload?.warnings.join(" · ")} /> : null}
      {items.length > 0 && payload?.warnings.length ? <DataState title="Fuentes parciales" detail={payload.warnings.join(" · ")} /> : null}
    </div>
  );
}
