# Bot económico AR

Bot de Telegram para consultar cotizaciones de mercado, reservas del BCRA y recibir resúmenes diarios.

## Arquitectura

El código está organizado en paquetes modulares dentro de `bot_econ/`:

- `config`: carga de variables de entorno y configuración de logging estructurado en JSON.
- `formatters`: formateadores reutilizables para números, monedas y porcentajes.
- `data_sources`: clientes asíncronos con caché TTL y reintentos para cada proveedor externo.
- `services`: lógica de negocio (persistencia en Redis, migraciones, pipelines de métricas).
- `telegram`: handlers desacoplados de la API de Telegram, con scheduler para precalentar datos.
- `main.py`: punto de entrada que inicializa configuración, servicios y arranca el bot.

## Requisitos

- Python 3.11+
- Redis accesible mediante `REDIS_URL`
- Token del bot en la variable `TELEGRAM_TOKEN` (o `BOT_TOKEN`)

Dependencias principales en `requirements.txt`; herramientas de desarrollo en `requirements-dev.txt`.

## Configuración

Variables de entorno relevantes:

| Variable | Descripción |
| --- | --- |
| `TELEGRAM_TOKEN` | Token del bot. |
| `REDIS_URL` | Cadena de conexión a Redis. |
| `REDIS_PREFIX` | Prefijo opcional para namespacing. |
| `BASE_URL` | URL pública utilizada para webhooks. |
| `WEBHOOK_SECRET` | Segmento del path del webhook. |

## Ejecución local

1. Crear entorno virtual y instalar dependencias:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Lanzar Redis (por ejemplo con Docker):

   ```bash
   docker run -p 6379:6379 redis:7
   ```

3. Exportar variables de entorno mínimas:

   ```bash
   export TELEGRAM_TOKEN=... REDIS_URL=redis://localhost:6379/0
   ```

4. Ejecutar el bot:

   ```bash
   python -m bot_econ.main
   ```

## Calidad y pruebas

- `pytest` con fixtures para validar migraciones y formateadores.
- `ruff`, `black` y `mypy` integrados vía GitHub Actions (`.github/workflows/ci.yml`).

Comandos recomendados:

```bash
ruff check bot_econ
black --check bot_econ
mypy bot_econ
pytest
```

## Despliegue

Render u otras plataformas pueden utilizar `render.yaml` como referencia. Para entornos contenedorizados se recomienda crear un `Dockerfile` multi-stage (pendiente de contribución) que valide la presencia de variables críticas al inicio.

