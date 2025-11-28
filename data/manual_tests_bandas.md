# Pruebas manuales de `format_bandas_cambiarias`

Ejecutar los siguientes snippets dentro del repo con `python - <<'PY'` para validar el formateo. 
Si el módulo exige token de Telegram, exportar previamente `TELEGRAM_TOKEN=dummy` (o un token real).

## Caso 1: variación superior con signo invertido (simulado)
```
from bot_econ_full_plus_rank_alerts import format_bandas_cambiarias
sample = {
    "banda_superior": 1055.5,
    "banda_inferior": 864.4,
    "variacion_superior": -0.38,  # viene invertida
    "variacion_inferior": -0.38,
    "variacion_diaria": 0.38,
    "fecha": "2024-11-04 12:00:00",
}
print(format_bandas_cambiarias(sample))
```
La banda superior debe mostrarse en verde con flecha hacia arriba (0,38%).

## Caso 2: datos mensuales reales (ejemplo 2024-10-31, variación en baja)
```
from bot_econ_full_plus_rank_alerts import format_bandas_cambiarias
realistic = {
    "banda_superior": 1014.12,
    "banda_inferior": 831.49,
    "variacion_mensual_superior": -1.1,
    "variacion_mensual_inferior": -1.1,
    "fecha": "2024-10-31 17:00:00",
}
print(format_bandas_cambiarias(realistic))
```
La banda superior debería reflejar la baja con ícono rojo hacia abajo.
