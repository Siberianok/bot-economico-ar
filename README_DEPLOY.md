
# Bot Económico AR — Deploy en Render (plan Free)

## 1) Archivos
- `bot_econ_full_plus_rank_alerts.py` — script principal del bot (worker).
- `requirements.txt` — dependencias.
- `render.yaml` — blueprint de Render (servicio tipo Worker).

## 2) Variables de entorno
- `BOT_TOKEN` — el token del bot de Telegram (NO hardcodear).  
- (opcional) `ALERTS_DB_PATH` — ruta del SQLite para alertas (por defecto `./alerts.db`). En Render esta guía usa `/var/tmp/alerts.db` para minimizar pérdidas por redeploy.

## 3) Deploy rápido (desde un repo)
1. Subí estos 3 archivos a un repositorio (GitHub/GitLab).
2. En Render: **New +** → **Blueprint** → conectá el repo.
3. Aceptá el servicio **worker** sugerido por `render.yaml`.
4. En *Environment variables*, agregá `BOT_TOKEN` con tu token real.
5. Deploy.

> Plan Free: Render puede reiniciar el worker de vez en cuando. Las alertas se persisten en SQLite local; si el contenedor se reinicia, el archivo en `/var/tmp` suele sobrevivir al *restart* del proceso, pero no a un *redeploy*. Para persistencia “fuerte”, migrá a una pequeña base externa (p.ej. PostgreSQL Free Tier) con `sqlite3` → `psycopg2` (cambios mínimos).

## 4) Comandos finales
- `/dolar` — Tipos de cambio (Blue, MEP, CCL, Cripto, Oficial, Mayorista).  
  Fuente principal: DolarAPI (Ámbito); fallback: CriptoYa.
- `/reservas` — Reservas BCRA (series oficiales via `apis.datos.gob.ar`).
- `/inflacion` — Inflación (variación mensual, INDEC via `apis.datos.gob.ar`).
- `/riesgo` — Riesgo país (ArgentinaDatos).
- `/acciones` — Top 3 BYMA por rendimiento 3M (Yahoo Finance).
- `/cedears` — Top 3 CEDEARs por rendimiento 3M.
- `/ranking_acciones` — Top 5 BYMA por proyección 6M (0.1·1m + 0.3·3m + 0.6·6m).
- `/ranking_cedears` — Top 5 CEDEARs por proyección 6M.
- `/alerta_dolar <tipo> <umbral>` — Crea alerta (blue|mep|ccl|cripto|oficial|mayorista).
- `/alertas` — Lista alertas activas.
- `/alerta_borrar <id>` — Elimina una alerta.
- `/resumen_diario` — Dólares + Reservas + Inflación + Riesgo + 5 noticias filtradas.

## 5) Notas técnicas
- Yahoo Finance: se usa endpoint `v8/finance/chart` (sin API key) con 6M/1D; retornos 1m/3m/6m se calculan sobre ~21/63/126 ruedas.
- `apis.datos.gob.ar`: se prueban múltiples IDs de series para robustez; si cambia el ID oficial, actualizalo en las constantes del script.
- Noticias: parser RSS mínimo + filtro de keywords (evita salud, quiniela, etc.). Podés editar listas `NEWS_KEYWORDS_POS/NEG` para afinar.

## 6) Pruebas locales
```bash
python -m venv .venv && source .venv/bin/activate  # (en Windows: .venv\Scripts\activate)
pip install -r requirements.txt
export BOT_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python bot_econ_full_plus_rank_alerts.py
```
