# Deployment Render

Este repo queda preparado para deploy real en Render con tres servicios separados:

- `bot-economico-ar`: bot Telegram legacy.
- `observatorio-economico-api`: backend FastAPI.
- `observatorio-economico-miniapp`: frontend estático HTTPS.

No se deben cargar tokens ni secretos en el código. Las variables marcadas como `sync: false` se configuran manualmente en Render.

## 1. Blueprint

Crear el Blueprint desde `render.yaml`.

Servicios esperados:

| Servicio | Tipo | Build command | Start / Publish |
| --- | --- | --- | --- |
| `bot-economico-ar` | Web Service Python | `pip install -r requirements.txt` | `python main.py` |
| `observatorio-economico-api` | Web Service Python | `pip install -r requirements.txt` | `python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT` |
| `observatorio-economico-miniapp` | Static Site | `npm install && npm run build` | `dist` |

El frontend usa `rootDir: frontend` y rewrite SPA hacia `/index.html`.

## 2. Variables

### Bot

- `TELEGRAM_TOKEN`: token real del bot.
- `WEBHOOK_SECRET`: path secreto del webhook.
- `MINIAPP_URL`: URL HTTPS final del Static Site.
- `UPSTASH_REDIS_REST_URL`: URL REST de Upstash, si se usa.
- `UPSTASH_REDIS_REST_TOKEN`: token REST de Upstash, si se usa.
- `UPSTASH_STATE_KEY`: clave de estado, opcional.

### Backend API

- `DATABASE_URL`: por defecto `sqlite:///./data/observatorio.db`.
- `API_CORS_ORIGINS`: URL HTTPS final del frontend, separada por coma si hay más de un origen.
- `MINIAPP_URL`: URL HTTPS final del Static Site.
- `SIGNAL_MIN_SCORE`: default `70`.
- `LEGACY_LIVE_FETCH_ENABLED`: default `0`; usar `1` solo si se quieren consultas live legacy.
- `DEV_TELEGRAM_USER_ID`: usuario local/dev opcional.

### Frontend

- `VITE_API_BASE_URL`: URL HTTPS final del backend API.
- `VITE_MINIAPP_ENV`: `production`.

## 3. Orden Recomendado

1. Crear Blueprint.
2. Cargar secretos del bot.
3. Desplegar `observatorio-economico-api`.
4. Copiar su URL pública HTTPS.
5. Configurar `VITE_API_BASE_URL` en el Static Site.
6. Desplegar `observatorio-economico-miniapp`.
7. Copiar su URL pública HTTPS.
8. Configurar `MINIAPP_URL` en el bot y en la API.
9. Configurar `API_CORS_ORIGINS` con la URL HTTPS del frontend.
10. Redeploy del bot y de la API después de ajustar variables.

## 4. Health Checks

Backend:

- `https://BACKEND_RENDER_URL/api/v1/health`
- `https://BACKEND_RENDER_URL/docs`

Frontend:

- `https://FRONTEND_RENDER_URL`

Bot:

- En Telegram, enviar `/dashboard`.
- Tocar `Abrir Dashboard`.
- Verificar que abre dentro de Telegram en celular.

## 5. Pruebas Post-Deploy

1. Abrir `/api/v1/health` y confirmar `status: "ok"`.
2. Abrir `/docs` y confirmar endpoints.
3. Abrir el frontend Render en navegador.
4. Revisar que Settings muestre la URL del backend configurada.
5. En Telegram celular, ejecutar `/dashboard`.
6. Tocar `Abrir Dashboard`.
7. Revisar pantallas: Market Pulse, Screener, Portfolio Command Center, Alerts, Signals y Settings.

## 6. Límites Actuales

- SQLite en Render sin disco persistente puede perder estado en redeploy, restart o recreación del servicio.
- Para producción estable, migrar `DATABASE_URL` a PostgreSQL administrado.
- Market live sigue desactivado salvo `LEGACY_LIVE_FETCH_ENABLED=1`.
- Portfolio, validaciones financieras avanzadas y partes no adaptadas siguen respondiendo `not_available`.
