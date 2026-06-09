from app.registry.analyzers import list_analyzers
from app.registry.analyzers import resolve as resolve_analyzer
from app.registry.steg import list_models
from app.registry.steg import resolve as resolve_steg


def test_steg_registry_has_ten_models():
    models = list_models()
    assert len(models) == 10
    for m in models:
        assert resolve_steg(m.key).key == m.key


def test_analyzer_registry_has_ten():
    analyzers = list_analyzers()
    assert len(analyzers) == 10
    for a in analyzers:
        assert resolve_analyzer(a.key).key == a.key


def test_unknown_keys_raise():
    import pytest

    with pytest.raises(KeyError):
        resolve_steg("nope")
    with pytest.raises(KeyError):
        resolve_analyzer("nope")
