# Mini App: ejecución local y Telegram

## Backend local sin Telegram

```powershell
$env:TELEGRAM_RUNTIME_ENABLED="0"
$env:LEGACY_LIVE_FETCH_ENABLED="1"
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Abrir:

- `http://127.0.0.1:8000/api/v1/health`
- `http://127.0.0.1:8000/docs`

## Frontend local

Crear `frontend/.env.local` sin subirlo al repositorio:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_MINIAPP_ENV=development
VITE_DEV_TELEGRAM_USER_ID=local-dev
```

Ejecutar:

```powershell
cd frontend
npm install
npm run dev
```

Abrir `http://127.0.0.1:5173`.

## Runtime combinado local

No usar el token de producción para una prueba innecesaria. Cuando se quiera validar expresamente el
webhook combinado se requieren:

```powershell
$env:TELEGRAM_RUNTIME_ENABLED="1"
$env:TELEGRAM_TOKEN="..."
$env:WEBHOOK_SECRET="..."
$env:BASE_URL="https://URL-HTTPS-PUBLICA"
$env:MINIAPP_URL="https://URL-MINIAPP"
```

Telegram no acepta `localhost` como webhook. La prueba completa se realiza después del deploy HTTPS o
mediante un túnel temporal aprobado para esa validación.

## Integración Telegram

La Mini App detecta `window.Telegram.WebApp`, llama `ready()` y `expand()`, y envía
`X-Telegram-Init-Data` al backend cuando está disponible. En producción el backend valida firma,
`auth_date` y usuario con `TELEGRAM_TOKEN`; el valor no debe aceptarse sin validación. En desarrollo,
con `TELEGRAM_AUTH_REQUIRED=0`, usa el usuario configurado por `VITE_DEV_TELEGRAM_USER_ID`.

`GET /api/v1/session` permite comprobar desde la interfaz si la identidad fue validada por el
backend y si el usuario tiene un PF legacy disponible. El detalle de seguridad y cálculo se encuentra
en `docs/portfolio_v2.md`.

Prueba en celular:

1. Ejecutar `/start` y confirmar que se conservan los botones existentes.
2. Ejecutar `/dashboard`.
3. Tocar `Abrir Dashboard`.
4. Verificar Inicio, Economía, Screener, Noticias, Portfolio, Alertas, Lecturas automáticas y Ajustes.
5. Confirmar en Ajustes que `initData` está disponible y `Validación backend` figura como `Verificada`.
6. Comparar Portfolio con `/portafolio` y comprobar cantidad de posiciones, capital y cobertura.
7. Comparar la pestaña Telegram de Alertas con `/alertas_menu` y comprobar cantidad, estado y umbrales.
8. Comparar Suscripción con `/subs` y verificar el horario del resumen diario.
9. Crear un borrador de alerta y confirmar que queda separado de las reglas Telegram activas.

Si falta `MINIAPP_URL`, el mensaje esperado sigue siendo:

```text
Dashboard no configurado. Definí MINIAPP_URL.
```

## Estados de datos

La interfaz diferencia `loading`, `error`, `stale`, `fallback` y `not_available`. No se deben reemplazar
datos ausentes por valores simulados.

Con `LEGACY_LIVE_FETCH_ENABLED=1`, Market Pulse consulta tipos de cambio, reservas, inflacion,
riesgo pais y bandas mediante las fuentes reales ya integradas en el bot legacy. El Screener obtiene
precios, retornos, volatilidad y proyecciones para acciones y CEDEARs; en CEDEARs separa rendimiento
ARS, subyacente USD y efecto FX. Senales evalua reglas transparentes sobre esos datos reales y aplica
cooldown antes de persistir una nueva lectura. Los endpoints se mantienen como `partial` mientras
liquidez, noticias o bloques historicos no tengan una adaptacion confiable.

Portfolio usa un adaptador de solo lectura sobre una copia profunda de `PF`. Solo devuelve posiciones
cuando la solicitud incluye una identidad Telegram numerica validada y ese usuario tiene un portfolio
legacy cargado en el proceso. Fuera de Telegram, el usuario de desarrollo `local-dev` recibe
`not_available`; el backend nunca elige ni expone el portfolio de otro chat como fallback. El adaptador
no llama `load_state()`, `save_state()` ni `pf_market_snapshot()` y no modifica `PF`.

En modo local fuera de Telegram, Settings muestra el SDK cargado pero `Contexto Telegram: modo
desarrollo`. La deteccion positiva exige `initData`, usuario Telegram o una plataforma WebApp real.

Noticias consume en modo de solo lectura los RSS configurados en el bot legacy. `/api/v1/news`
devuelve únicamente artículos recibidos de esas fuentes y `/api/v1/summary` compone el inicio con
mercado, noticias y lecturas automáticas. Una fuente que falla queda documentada en `warnings`; no se
rellena con titulares inventados.

La correspondencia completa entre comandos y pantallas está documentada en
`docs/miniapp_menu_parity.md`.

Alertas muestra `ALERTS` y `SUBS` del usuario autenticado mediante copias profundas de solo lectura.
Las reglas Telegram operativas se mantienen separadas de los borradores SQLite, que continúan
aislados por usuario y con delivery `not_connected`. La promoción de un borrador y cualquier cambio
de estado legacy requieren una fase posterior con persistencia externa y auditoría de escritura.
