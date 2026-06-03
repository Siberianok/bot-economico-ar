import { FormEvent, useState } from "react";
import { createAlert, getAlerts } from "../api/alerts";
import { StatusBadge } from "../components/Badges/StatusBadge";
import { DataState } from "../components/States/DataState";
import { DataTable } from "../components/Tables/DataTable";
import { useEnvelope } from "../hooks/useEnvelope";

export function Alerts() {
  const [refresh, setRefresh] = useState(0);
  const [asset, setAsset] = useState("dolar_cripto");
  const [target, setTarget] = useState("1420");
  const [condition, setCondition] = useState<"exact" | "approximation">("exact");
  const { payload, loading, error } = useEnvelope(getAlerts, [refresh]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    await createAlert({
      asset,
      target_value: Number(target),
      condition_type: condition,
      operator: ">=",
      prealert_tolerance_pct: condition === "approximation" ? 0.5 : null,
      cooldown_minutes: 15,
    });
    setRefresh((value) => value + 1);
  }

  const rows =
    payload?.data?.items?.map((item) => [
      item.asset,
      item.condition_type,
      `${item.operator} ${item.target_value}`,
      `${item.cooldown_minutes} min`,
      item.trigger_count,
      <StatusBadge status={item.status} />,
    ]) || [];

  return (
    <div className="space-y-5">
      <header>
        <div className="text-xs uppercase tracking-wide text-obs-blue">Alertas</div>
        <h2 className="text-3xl font-semibold">Reglas y disparos</h2>
      </header>
      <form className="grid gap-3 rounded-lg border border-obs-border bg-obs-card p-4 md:grid-cols-5" onSubmit={submit}>
        <input className="field" value={asset} onChange={(e) => setAsset(e.target.value)} aria-label="Activo" />
        <input className="field" value={target} onChange={(e) => setTarget(e.target.value)} aria-label="Objetivo" />
        <select className="field" value={condition} onChange={(e) => setCondition(e.target.value as any)}>
          <option value="exact">Exacta</option>
          <option value="approximation">Aproximación</option>
        </select>
        <div className="rounded-md border border-obs-border bg-obs-card2 px-3 py-2 text-sm text-obs-muted">Cooldown 15m</div>
        <button className="rounded-md bg-obs-blue px-4 py-2 text-sm font-semibold text-obs-bg" type="submit">
          Crear alerta
        </button>
      </form>
      {loading ? <DataState title="Cargando alertas" /> : null}
      {error ? <DataState title="Error de carga" detail={error} /> : null}
      {payload ? (
        rows.length ? (
          <DataTable headers={["Activo", "Tipo", "Condición", "Cooldown", "Disparos", "Estado"]} rows={rows} />
        ) : (
          <DataState title="Sin alertas creadas" detail="Usá el formulario para crear una alerta exacta o de aproximación." />
        )
      ) : null}
    </div>
  );
}

