# Mini App Setup

## Desarrollo Local

Backend:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m uvicorn backend.app.main:app --reload
```

Abrir:

- `http://127.0.0.1:8000/api/v1/health`
- `http://127.0.0.1:8000/docs`

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Abrir:

- `http://127.0.0.1:5173`

Variables locales recomendadas:

- `VITE_API_BASE_URL=http://127.0.0.1:8000`
- `VITE_MINIAPP_ENV=development`
- `VITE_DEV_TELEGRAM_USER_ID=local-dev`
- `MINIAPP_URL=http://127.0.0.1:5173`
- `API_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173`
- `DATABASE_URL=sqlite:///./data/observatorio.db`
- `LEGACY_LIVE_FETCH_ENABLED=0`

## Producción Render

El frontend se despliega como Static Site:

- `rootDir`: `frontend`
- `buildCommand`: `npm install && npm run build`
- `staticPublishPath`: `dist`
- rewrite SPA: `/* -> /index.html`

Configurar en Render:

- `VITE_API_BASE_URL=https://BACKEND_RENDER_URL`
- `VITE_MINIAPP_ENV=production`

El backend debe permitir el origen del frontend:

- `API_CORS_ORIGINS=https://FRONTEND_RENDER_URL`

El bot debe conocer la Mini App:

- `MINIAPP_URL=https://FRONTEND_RENDER_URL`

No hardcodear estas URLs en el código. Se cargan como variables de entorno cuando Render genere las URLs reales.

## Telegram

1. Confirmar que `MINIAPP_URL` está cargada en el servicio del bot.
2. Redeploy del bot.
3. Enviar `/dashboard` en Telegram.
4. Tocar `Abrir Dashboard`.
5. Verificar que abre dentro de Telegram en celular.

Si `MINIAPP_URL` no está configurada, el bot responde:

```text
Dashboard no configurado. Definí MINIAPP_URL.
```

## Endpoints Para Probar

- `GET /api/v1/health`
- `GET /api/v1/market/pulse`
- `GET /api/v1/screener?kind=acciones`
- `GET /api/v1/screener?kind=cedears`
- `GET /api/v1/portfolio/summary`
- `GET /api/v1/alerts`
- `GET /api/v1/signals`
- `GET /api/v1/config`
- `GET /api/v1/projections`
- `GET /api/v1/validations`

## Advertencias

- SQLite en Render sin disco persistente puede perder datos entre reinicios o redeploys.
- Para uso productivo sostenido, migrar a PostgreSQL.
- Las secciones no adaptadas del legacy deben seguir apareciendo como `not_available`.
- `LEGACY_LIVE_FETCH_ENABLED=1` habilita consultas reales legacy y debe activarse solo cuando el entorno esté listo.
