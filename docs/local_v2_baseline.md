# Línea base local ObeconBot V2

Fecha: 2026-08-27.

La fuente de trabajo V2 es `C:\Users\ariel\Downloads\IA\ObeconBot`.

Estado al iniciar esta fase:

- La carpeta no contiene `.git`; todavía no está reconciliada con GitHub.
- Render y GitHub no fueron modificados.
- Los servicios Render existentes deben permanecer suspendidos durante el desarrollo local.
- El deploy objetivo es un Web Service combinado más un Static Site.

Validación mínima antes de sincronizar o desplegar:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip_audit -r requirements.txt -r requirements-dev.txt --progress-spinner=off
cd frontend
npm ci
npm run build
npm audit --audit-level=moderate
```

Una sincronización GitHub o una modificación Render requiere una autorización independiente.

## Estado validado al 2026-09-01

- Import de `backend.app.main`: correcto.
- Suite completa: `126 passed`.
- Quality gate: modo `Blocking`, 10 pasos obligatorios, sin fallos ni warnings.
- Frontend: typecheck y build de producción correctos; las pantallas se cargan por módulos y ECharts
  queda aislado en un chunk opcional.
- Auditoría de dependencias: `pip-audit` y `npm audit` sin vulnerabilidades conocidas.
- Validación visual: seis pantallas recorridas en desktop y mobile sin overflow horizontal ni errores
  funcionales en el estado final renderizado.
- Market Pulse y Screener consumen fuentes reales adaptadas desde el legacy.
- Señales aplica reglas transparentes sobre Market Pulse con deduplicación por cooldown.
- Portfolio dispone de un adaptador de solo lectura por identidad Telegram; fuera de Telegram no
  expone datos de otro usuario y responde `not_available`.
- Render y GitHub continúan sin modificaciones.

## Reconciliación web al 2026-09-01

- GitHub `Siberianok/bot-economico-ar` conserva una revisión anterior a la V2 local. Su último push
  observado es del 2026-06-03 y todavía declara tres servicios Render separados.
- Render contiene dos grupos duplicados de esos tres servicios: uno con nombres base y otro con
  sufijo `-jvcf`. Ninguno se modifica ni elimina desde esta fase.
- Ambas APIs públicas responden en `/api/v1/health`, pero corresponden al contrato anterior y no
  exponen `/readyz` ni el runtime Telegram combinado.
- Ambas Mini Apps públicas cargan el HTML, pero fueron compiladas sin `VITE_API_BASE_URL`. Además,
  las APIs no autorizan sus orígenes CORS. Por lo tanto no son candidatas de aceptación V2.
- La V2 local responde en backend y frontend, pasa los tests y el build, pero debe sincronizarse con
  una fuente Git canónica antes de crear el staging HTTPS para Telegram.

Deuda controlada:

- `bot_econ_full_plus_rank_alerts.py` conserva observaciones de lint históricas. No se aplica un
  formateo masivo para evitar una reescritura riesgosa del bot estable.
- `.packages`, `.pytest-tmp`, `.pytest-basetemp-render-dashboard`, `pytest-cache-files-*` y
  `data/.pytest-resume` son residuos locales ignorados. Windows mantiene ACL que impiden eliminarlos
  desde esta sesión; no son importados, empaquetados ni necesarios para ejecutar el proyecto.
- La carpeta aún no contiene `.git`; reconciliarla con GitHub sigue siendo requisito previo al deploy.
