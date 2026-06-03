def test_backend_imports_without_legacy_side_effects():
    import backend
    import backend.app.main
    from backend.app.main import app

    assert backend is not None
    assert app.title

