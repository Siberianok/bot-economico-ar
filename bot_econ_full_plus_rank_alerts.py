"""Compatibilidad con despliegues legacy.

Render ejecuta este módulo directo (`python bot_econ_full_plus_rank_alerts.py`),
pero la lógica actual vive en :mod:`bot_econ.main`. Este wrapper delega en el
nuevo entrypoint para evitar errores de "file not found".
"""

from bot_econ.main import main


if __name__ == "__main__":
    main()
