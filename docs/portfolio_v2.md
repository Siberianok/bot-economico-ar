# Portfolio V2: integración Telegram de solo lectura

## Flujo de identidad

1. Telegram abre la Mini App desde `/dashboard`.
2. El frontend lee `window.Telegram.WebApp.initData`.
3. Cada solicitud envía `X-Telegram-Init-Data` al backend.
4. El backend valida firma, `auth_date`, tamaño, campos duplicados y usuario positivo usando
   `TELEGRAM_TOKEN`.
5. El ID firmado se usa como única clave para consultar `PF`.

El cliente no puede indicar otro `chat_id`. En desarrollo, `local-dev` se identifica expresamente
como fallback no verificado y no puede leer el PF de un usuario Telegram.

## Estado legacy

En producción, `obeconbot-core` ejecuta el runtime combinado. Durante el arranque, el runtime carga
el estado legacy desde Upstash o el fallback configurado y deja `PF` disponible en el mismo proceso.
El adaptador:

- consulta únicamente `PF[user_id]`;
- usa `deepcopy` antes de procesar;
- no llama `load_state()`, `save_state()` ni `pf_market_snapshot()`;
- no modifica base, instrumentos, cotizaciones cacheadas ni historial.

## Contratos

- `GET /api/v1/session`: indica si la identidad fue validada por el servidor, el estado del runtime
  y si existe un PF para ese usuario.
- `GET /api/v1/portfolio/summary`: devuelve configuración, capital invertido, valor actual,
  resultado, cobertura, posiciones, exposición monetaria, aportantes/detractores, proyecciones,
  volatilidad, benchmark y métricas parciales.

TWR y MWR se etiquetan como `partial` porque son aproximaciones de la foto actual. El cálculo
canónico, drawdown y rebalanceo requieren una serie histórica de valuaciones confiable.

## Criterios de aceptación en Telegram

- Ajustes muestra `Validación backend: Verificada`.
- Portfolio muestra el nombre del usuario y la cantidad de posiciones legacy.
- Los totales coinciden con `/portafolio` para los instrumentos con precio observado.
- Una posición sin precio queda fuera del total actual y reduce la cobertura; no se inventa valor.
- Recargar la Mini App no modifica PF ni genera escrituras en Upstash.
