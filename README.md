# ObeconBot

Observatorio económico argentino con Telegram Bot, API FastAPI y Mini App React.

## Desarrollo local

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

En otra terminal:

```powershell
cd frontend
npm run dev
```

- API: `http://127.0.0.1:8000/docs`
- Mini App: `http://127.0.0.1:5173`

La arquitectura, variables y despliegue están documentados en `docs/`.
