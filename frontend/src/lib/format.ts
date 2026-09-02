export function formatObservationDate(value: unknown): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;

  const raw = value.trim();
  const dateOnly = /^\d{4}-\d{2}-\d{2}$/.test(raw);
  const normalized = dateOnly ? `${raw}T12:00:00` : raw.replace(" ", "T");
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return raw;

  return new Intl.DateTimeFormat("es-AR", {
    dateStyle: "short",
    ...(dateOnly ? {} : { timeStyle: "short" as const }),
  }).format(parsed);
}
