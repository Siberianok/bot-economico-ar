# Bot Económico AR — Deploy en Render (plan Free)

## 1) Archivos
- `bot_econ/` — paquete del bot con módulos `config`, `data_sources`, `services`, `telegram` y `main.py`.
- `requirements.txt` — dependencias runtime.
- `requirements-dev.txt` — herramientas de calidad.
- `render.yaml` — blueprint de Render.

## 2) Variables de entorno
- `TELEGRAM_TOKEN` — token del bot (obligatorio).
- `REDIS_URL` — conexión a Redis (obligatorio).
- `REDIS_PREFIX` — prefijo opcional (default `bot-ar`).
- `BASE_URL`, `WEBHOOK_SECRET`, `PORT` — si se usan webhooks. Para polling no son necesarios.

## 3) Deploy rápido (desde un repo)
1. Subí estos archivos a un repositorio (GitHub/GitLab).
2. En Render: **New +** → **Blueprint** → conectá el repo.
3. Aceptá el servicio sugerido por `render.yaml`.
4. Configurá las variables de entorno (`TELEGRAM_TOKEN`, `REDIS_URL`, etc.).
5. Deploy.

## 4) Comandos disponibles
- `/start` — mensaje de bienvenida.
- `/resumen` — cotizaciones oficiales/digitales y reservas BCRA.
- `/healthcheck` — ping a Redis.

> El pipeline modular permite agregar fácilmente nuevas fuentes (riesgo país, métricas bursátiles, alertas, etc.) sin tocar los handlers existentes.

## 5) Notas técnicas
- Todas las llamadas HTTP usan una única `ClientSession`, caché TTL y reintentos exponenciales.
- La persistencia usa Redis de forma asíncrona. `StateService` migra automáticamente estados legacy `state.json` la primera vez.
- El `JobQueue` precalienta datos cada 5 minutos para reducir la latencia percibida por los usuarios.

## 6) Pruebas locales
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
export TELEGRAM_TOKEN=xxxx REDIS_URL=redis://localhost:6379/0
python -m bot_econ.main
```
