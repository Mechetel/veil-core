from celery import Celery

from app.config import settings

celery = Celery(
    "veil_core",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.tasks.steganography",
        "app.tasks.steganalysis",
    ],
)

celery.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    # Each domain has its own queue + worker role.
    task_routes={
        "app.tasks.steganography.*": {"queue": "steg"},
        "app.tasks.steganalysis.*": {"queue": "analysis"},
    },
)
