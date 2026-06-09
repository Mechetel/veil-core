import pytest

from app.celery_app import celery

# Ensure tasks are registered for eager execution.
import app.tasks.steganography  # noqa: E402,F401
import app.tasks.steganalysis  # noqa: E402,F401


@pytest.fixture(autouse=True)
def eager_celery():
    celery.conf.task_always_eager = True
    celery.conf.task_eager_propagates = False
    yield
    celery.conf.task_always_eager = False


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture
def auth():
    from app.config import settings

    return {"X-Auth-Token": settings.veil_core_token}
