# Paridad entre Telegram y la Mini App

La Mini App complementa al bot actual. No reemplaza ni migra el estado legacy de
ALERTS, SUBS o PF en esta fase.

| Comando o menu Telegram | Superficie Mini App | Estado actual |
| --- | --- | --- |
| `/start`, `/menu`, `/resumen` | Inicio | Resumen real de mercado, accesos rapidos, noticias y lecturas automaticas. |
| `/economia`, `/reservas`, `/inflacion`, `/riesgo` | Mercado > Economia | Datos reales cuando las fuentes legacy responden; cada bloque informa su frescura. |
| `/noticias` | Noticias | Lectura real y de solo lectura de los RSS configurados en el bot. |
| `/acciones` | Screener > Acciones | Ranking y rendimientos reales cuando existe cotizacion confiable. |
| `/cedears` | Screener > CEDEARs | Ranking ARS/subyacente/FX cuando las fuentes permiten separar los componentes. |
| `/portafolio` | Portfolio | Resumen, posiciones, analitica y gestion. Requiere una identidad Telegram validada para leer PF legacy. |
| `/alertas_menu`, `/alertas_pause` | Alertas | Reglas nuevas persistidas en SQLite. La entrega por Telegram y las alertas legacy siguen separadas. |
| `/subs` | Mas | Continúa disponible en Telegram; SUBS no se migra en esta fase. |
| `/performance`, `/performance_detalle` | Mas | Continúan disponibles en Telegram; la vista detallada no se replica todavia. |
| `/dashboard` | Entrada a la Mini App | Abre el dashboard cuando `MINIAPP_URL` esta configurada. |

## Lecturas automaticas

Las lecturas automaticas son condiciones transparentes derivadas de valores de
mercado observados. Muestran activo, metrica, score, severidad y fuente. No son
recomendaciones de inversion y no se crean cuando faltan datos confiables.

## Limites deliberados

- PF, ALERTS y SUBS no se copian ni alteran.
- El portfolio local de desarrollo queda como `not_available` si no existe una
  identidad Telegram numerica y validada.
- La agenda economica permanece `not_available` hasta integrar una fuente
  verificable.
- Las alertas creadas en la Mini App son borradores locales; el canal Telegram
  se muestra como `not_connected` hasta implementar autenticacion y entrega.
