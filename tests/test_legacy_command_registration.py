import re
from pathlib import Path


def test_legacy_commands_are_registered_once_or_more_as_expected():
    source = Path("bot_econ_full_plus_rank_alerts.py").read_text(encoding="utf-8")
    commands = {
        "start",
        "menu",
        "dashboard",
        "economia",
        "acciones",
        "cedears",
        "portafolio",
        "alertas_menu",
        "subs",
        "resumen",
        "performance",
        "performance_detalle",
    }

    for command in commands:
        pattern = rf'CommandHandler\(\s*"{command}"'
        assert re.search(pattern, source), f"Falta registrar /{command}"

    assert len(re.findall(r'CommandHandler\(\s*"performance_detalle"', source)) == 1
