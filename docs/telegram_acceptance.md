# Aceptación Telegram de ObeconBot V2

## Objetivo

Validar la V2 dentro de Telegram sin reemplazar ni borrar los servicios anteriores. El bot conserva
la conversación por comandos y la Mini App agrega una vista visual; una modalidad no sustituye a la
otra.

## Alcance del staging

- Un Web Service temporal o candidato llamado `obeconbot-core` para FastAPI, webhook y runtime del
  bot.
- Un Static Site llamado `obeconbot-miniapp` para la interfaz React.
- El mismo bot de Telegram solo debe apuntarse al staging durante la ventana de prueba acordada. Un
  token no puede mantener polling o webhook activo en dos runtimes simultáneos.
- No eliminar, suspender ni reconfigurar los Blueprints anteriores hasta completar la aceptación.

## Configuración requerida

En `obeconbot-core`:

- `TELEGRAM_RUNTIME_ENABLED=1`
- `TELEGRAM_AUTH_REQUIRED=1`
- `TELEGRAM_TOKEN` cargado manualmente
- `WEBHOOK_SECRET` cargado manualmente
- `BASE_URL=https://URL-CORE`
- `MINIAPP_URL=https://URL-MINIAPP`
- `API_CORS_ORIGINS=https://URL-MINIAPP`
- credenciales Upstash actuales, sin copiarlas al repositorio

En `obeconbot-miniapp`:

- `VITE_API_BASE_URL=https://URL-CORE`
- `VITE_MINIAPP_ENV=production`

## Prueba previa desde navegador

1. Confirmar `200` en `/healthz` y `/api/v1/health`.
2. Confirmar `200` en `/readyz` y estado Telegram `running`.
3. Abrir la Mini App HTTPS y comprobar que Ajustes muestra la URL real de la API.
4. Confirmar que no aparece `VITE_API_BASE_URL no configurado`.
5. Confirmar que el backend acepta el origen exacto de la Mini App.

## Prueba conversacional en Telegram

Los comandos siguientes deben continuar respondiendo en el chat, sin depender de la Mini App:

- `/start`
- `/menu`
- `/economia`
- `/acciones`
- `/cedears`
- `/portafolio`
- `/alertas_menu`
- `/subs`
- `/resumen`
- `/performance`
- `/performance_detalle`

Se debe comparar una muestra de cotizaciones, inflación, riesgo país y portfolio con la respuesta del
bot anterior. La prueba valida formato, disponibilidad y procedencia; no exige que un valor de mercado
cambiante coincida con una captura vieja.

## Prueba de la Mini App en Telegram

1. Ejecutar `/dashboard` y tocar `Abrir Dashboard`.
2. Confirmar que Telegram abre la aplicación embebida, no un navegador externo.
3. Revisar Inicio, Economía, Screener, Noticias, Portfolio, Alertas, Señales y Ajustes.
4. En Ajustes, confirmar `initData: disponible`, validación backend `validated` y runtime `running`.
5. Comparar Portfolio con `/portafolio` para el mismo usuario.
6. Comparar reglas Telegram de Alertas con `/alertas_menu`.
7. Comparar Suscripción con `/subs`.
8. Confirmar que un borrador visual no modifica `ALERTS`, `SUBS` ni `PF` legacy.
9. Cerrar y volver a abrir la Mini App para verificar navegación y sesión.

## Criterio de aprobación

La V2 se aprueba cuando los comandos legacy siguen operativos, la identidad Telegram queda validada,
las pantallas muestran datos reales o `not_available`, y no se detectan escrituras sobre estado legacy
desde la Mini App. Después de la aprobación se decide por separado qué servicios anteriores conservar,
suspender o eliminar.

## Reversión

Si falla la prueba, se restaura el webhook al runtime anterior o se deshabilita el runtime del staging.
No se borra estado Upstash ni se modifica el token. Los servicios duplicados se mantienen hasta que la
reversión esté comprobada.
