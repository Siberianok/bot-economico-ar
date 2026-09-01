# Deploy gratuito en Render

> Estado actual: preparación local. No aplicar el Blueprint ni eliminar servicios existentes hasta
> reconciliar esta carpeta con GitHub y recibir autorización explícita.

La verificación pública del 2026-09-01 encontró dos Blueprints antiguos de tres servicios cada uno.
Las dos Mini Apps cargan, pero no tienen `VITE_API_BASE_URL` embebida y sus APIs rechazan los orígenes
CORS correspondientes. No deben reutilizarse como staging V2 sin una reconciliación explícita. El
protocolo de prueba está en `docs/telegram_acceptance.md`.

## Servicios objetivo

| Servicio | Tipo | Responsabilidad |
| --- | --- | --- |
| `obeconbot-core` | Web Service Free | FastAPI, webhook Telegram, jobs legacy y keep-alive |
| `obeconbot-miniapp` | Static Site | React/Vite y navegación SPA |

Las capas de bot y API siguen separadas en código, pero comparten una instancia Python para no duplicar
horas gratuitas.

## Variables de `obeconbot-core`

Obligatorias:

- `TELEGRAM_RUNTIME_ENABLED=1`
- `TELEGRAM_AUTH_REQUIRED=1`
- `TELEGRAM_INIT_DATA_MAX_AGE_SECONDS=3600`
- `TELEGRAM_TOKEN`: secreto cargado manualmente.
- `WEBHOOK_SECRET`: valor aleatorio cargado manualmente.
- `BASE_URL`: URL HTTPS final de `obeconbot-core`.
- `MINIAPP_URL`: URL HTTPS final de `obeconbot-miniapp`.
- `API_CORS_ORIGINS`: URL HTTPS exacta de `obeconbot-miniapp`.

Persistencia legacy:

- `UPSTASH_REDIS_REST_URL`
- `UPSTASH_REDIS_REST_TOKEN`
- `UPSTASH_STATE_KEY=bot-econ-state`

Opcionales:

- `KEEPALIVE_ENABLED=1`
- `DATABASE_URL=sqlite:///./data/obeconbot.db`
- `SIGNAL_MIN_SCORE=70`
- `LEGACY_LIVE_FETCH_ENABLED=0`

## Variables del Static Site

- `VITE_API_BASE_URL`: URL HTTPS final de `obeconbot-core`.
- `VITE_MINIAPP_ENV=production`.
- `NODE_VERSION=20.19.5` o una versión compatible con Vite 8.

## Orden futuro de despliegue

1. Convertir esta carpeta local en la fuente canónica y sincronizarla con GitHub.
2. Preservar las variables Upstash y Telegram del servicio actualmente desplegado.
3. Crear o actualizar `obeconbot-core` desde `render.yaml`.
4. Cargar manualmente secretos y URLs; nunca escribirlos en el repositorio.
5. Validar `https://CORE_URL/healthz`, `https://CORE_URL/readyz`,
   `https://CORE_URL/api/v1/health` y `https://CORE_URL/docs`.
6. Desplegar el Static Site con `VITE_API_BASE_URL=https://CORE_URL`.
7. Cargar `MINIAPP_URL=https://MINIAPP_URL` y `API_CORS_ORIGINS=https://MINIAPP_URL`.
8. Probar `/start`, `/dashboard` y todos los comandos legacy desde Telegram celular.
9. Solo después de validar, solicitar confirmación para eliminar Blueprints o servicios duplicados.

## Health y diagnóstico

`GET /api/v1/health` muestra el estado de API y Telegram:

- `disabled`: runtime desactivado por configuración.
- `configuration_error`: falta token o secreto.
- `running`: webhook y tareas Telegram iniciados.
- `error`: fallo de inicialización; revisar logs sin exponer secretos.

Render usa `GET /readyz` como health check: devuelve `503` cuando el runtime Telegram está habilitado
pero no llegó a `running`. `GET /healthz` solo confirma que el proceso FastAPI responde.

El webhook usa una URL fija `/telegram/webhook`. Telegram entrega `WEBHOOK_SECRET` en el header
`X-Telegram-Bot-Api-Secret-Token`; no se incluye el secreto en rutas ni logs.

El keep-alive consulta `BASE_URL/` cada cinco minutos. Un `200` en `/` y logs
`Keepalive ... -> 200` confirman su ejecución.

## Límites gratuitos

- Solo debe permanecer activo un Web Service gratuito en el workspace de ObeconBot.
- El comando de inicio usa un solo worker. No escalar a varias instancias hasta implementar un lease
  distribuido que garantice un único webhook y un único conjunto de jobs.
- SQLite es efímero en Render. Upstash conserva únicamente el estado legacy configurado para usarlo.
- Alertas y proyecciones creadas desde la Mini App son borradores locales y pueden perderse; su canal
  Telegram permanece `not_connected`.
- El keep-alive no evita la suspensión por agotar las 750 horas mensuales compartidas.
- Alertas y jobs no tienen garantía de tiempo real durante reinicios o suspensiones de plataforma.
