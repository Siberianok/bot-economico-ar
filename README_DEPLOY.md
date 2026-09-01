# Deploy de ObeconBot

La arquitectura gratuita vigente usa un Web Service combinado y un Static Site. Consultar:

- `docs/deployment.md`: Blueprint, variables, health checks y orden de despliegue.
- `docs/miniapp_setup.md`: ejecución local y prueba desde Telegram.
- `docs/architecture.md`: límites entre API, runtime Telegram y legacy.

No aplicar `render.yaml` sobre los servicios actuales hasta reconciliar la fuente local con GitHub y
preservar los secretos existentes.
