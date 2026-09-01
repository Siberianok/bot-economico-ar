# Arquitectura ObeconBot V2

## Objetivo

ObeconBot conserva tres capas lógicas y usa dos servicios físicos para entrar en la cuota gratuita:

1. Telegram Bot legacy, alojado dentro del proceso FastAPI.
2. API FastAPI modular bajo `/api/v1`.
3. Mini App React publicada como Static Site.

## Runtime combinado

`backend.app.main:app` es el único proceso web de producción. Su ciclo de vida:

1. Inicializa la base SQLAlchemy.
2. Si `TELEGRAM_RUNTIME_ENABLED=1`, importa el legacy de forma lazy.
3. Carga el estado legacy y registra sus jobs actuales.
4. Inicia `python-telegram-bot` sin polling ni servidor aiohttp propio.
5. Registra el webhook fijo en `/telegram/webhook` y valida
   `X-Telegram-Bot-Api-Secret-Token`.
6. Conserva los loops legacy de alertas, catálogo y keep-alive.
7. Al apagar, cancela las tareas, persiste estado y cierra la aplicación Telegram.

Con `TELEGRAM_RUNTIME_ENABLED=0`, la API inicia sin token y sin importar
`bot_econ_full_plus_rank_alerts.py`. El arranque legacy `python main.py` sigue disponible para
pruebas de compatibilidad, pero no es el comando recomendado para el deploy gratuito combinado.

## Persistencia

- `ALERTS`, `SUBS`, `PF` y estado legacy no se migran. En producción deben usar Upstash.
- SQLite mantiene las entidades nuevas de API durante desarrollo local.
- SQLite en Render es efímero y puede perderse en restart, spin-down o redeploy.
- Antes de considerar persistentes las alertas/proyecciones nuevas se debe adoptar una base externa.

## Contrato API

Todas las rutas `/api/v1` devuelven:

```json
{
  "status": "ok",
  "timestamp": "2026-08-27T00:00:00Z",
  "source": "service",
  "freshness": "current",
  "data": {},
  "warnings": []
}
```

Lo que no tenga una fuente fiable continúa como `not_available`.

Market Pulse y Screener reutilizan de forma lazy las consultas reales ya presentes en el legacy. El
servicio de noticias copia de forma segura el catálogo RSS legacy y lee esas fuentes sin ejecutar
handlers, polling ni escritura de estado. El resumen de Inicio combina Market Pulse y Noticias sin
ocultar los estados parciales de cada origen. El
servicio de senales aplica reglas deterministas sobre Market Pulse, persiste solo eventos disparados y
deduplica por cooldown. Portfolio consulta una copia de `PF` para el usuario Telegram autenticado y
calcula unicamente bloques con precio y conversion disponibles; no carga, guarda ni migra estado.
Alertas consulta copias profundas de `ALERTS[user_id]` y `SUBS[user_id]`, presenta las reglas
Telegram como operativas y mantiene los borradores SQLite en un bloque separado sin delivery.

## Seguridad y límites

- Tokens, secretos y URLs finales se cargan por variables de entorno.
- El secreto se valida con comparación constante antes de aceptar updates Telegram y nunca forma
  parte de la URL.
- La Mini App valida la firma y antigüedad de `initData`; cada recurso de usuario se consulta por el
  identificador Telegram firmado. El fallback local solo existe con autenticación desactivada.
- `/api/v1/session` expone el resultado de esa validación sin devolver `initData`, token ni secretos.
- Portfolio consulta una copia profunda de `PF[user_id]`; el adaptador no acepta un `chat_id`
  proporcionado por el cliente y no escribe estado legacy.
- Alertas y suscripciones aplican el mismo ownership firmado; su lectura no ejecuta helpers que
  puedan limpiar pausas, disparar reglas o persistir cambios.
- `/healthz` comprueba vida del proceso, `/readyz` exige que Telegram esté operativo cuando está
  habilitado y `/api/v1/health` expone el diagnóstico sin revelar secretos.
- Las alertas creadas por la Mini App son borradores aislados por usuario. No se presentan como
  entregables hasta conectar un runner y almacenamiento persistente.
- El despliegue combinado debe ejecutar un único worker y una sola instancia activa. Falta un lease
  distribuido para admitir escalado horizontal con seguridad.
- El keep-alive evita el idle mientras el proceso está activo, pero no evita agotamiento de cuota ni
  puede reactivar una instancia ya suspendida.
