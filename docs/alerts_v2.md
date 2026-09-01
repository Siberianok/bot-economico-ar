# Alertas y suscripciones V2

## Alcance de esta fase

La Mini App diferencia dos almacenes que no son equivalentes:

- `ALERTS` y `SUBS`: estado operativo del bot Telegram.
- SQLite: borradores creados desde la API, sin delivery Telegram.

`GET /api/v1/alerts` conserva `data.items` como alias de los borradores para
compatibilidad y agrega los bloques `summary`, `legacy`, `drafts` y
`subscription`.

## Lectura legacy

`LegacyAdapter.inspect_alert_state(user_id)`:

1. Acepta únicamente un identificador Telegram numérico validado por el backend.
2. Importa el módulo legacy de forma lazy.
3. Copia profundamente `ALERTS[user_id]` y `SUBS[user_id]`.
4. Lee pausas globales y defaults del motor sin ejecutar evaluaciones.
5. No llama `load_state()`, `save_state()`, `_is_rule_paused()` ni handlers.

Las pausas vencidas se interpretan como inactivas para la respuesta, pero no se
eliminan del diccionario legacy durante la consulta.

## Datos visibles

Para cada regla operativa se exponen tipo, activo, operador, objetivo, unidad,
estado, pausa, cooldown, tolerancia, armado y último disparo conocido. Las reglas
compuestas incluyen descripciones resumidas de sus condiciones, pero no exponen
snapshots internos.

La suscripción diaria muestra horario, zona `America/Argentina/Buenos_Aires` y
canal Telegram. Su gestión continúa en `/subs`; las reglas operativas continúan
administrándose con `/alertas_menu`.

## Límites actuales

- El legacy sólo conserva el último disparo por regla; no existe todavía un
  historial operativo completo.
- Los borradores SQLite no envían mensajes ni se promocionan automáticamente a
  `ALERTS`.
- En desarrollo, `local-dev` no puede consultar estados de usuarios Telegram y
  recibe `not_available`.
- La lectura real requiere el runtime combinado cargado y una sesión Mini App
  con `initData` válido.

## Próximo paso seguro

Agregar un historial append-only externo y un flujo explícito de promoción de
borrador a regla Telegram. Esa escritura deberá validar nuevamente ownership,
persistir en Upstash y registrar auditoría antes de habilitarse en producción.
