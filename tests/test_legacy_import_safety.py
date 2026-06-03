from backend.app.services.legacy_adapter import LegacyAdapter, legacy_adapter


def test_global_legacy_adapter_is_lazy_after_api_import():
    import backend.app.main  # noqa: F401

    assert legacy_adapter.import_error() is None


def test_new_legacy_adapter_starts_without_importing_legacy():
    adapter = LegacyAdapter()
    assert adapter.imported is False
    assert adapter.import_error() is None

