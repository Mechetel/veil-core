from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from the environment.

    Tokens
    ------
    ``VEIL_CORE_TOKEN``     — secret the web app sends in ``X-Auth-Token`` on every
                              request to this core API.
    ``VEIL_CALLBACK_TOKEN`` — secret this core sends in ``X-Auth-Token`` when it
                              POSTs job results back to the web callbacks.

    Wiring
    ------
    ``WEB_CALLBACK_URL`` — base URL of veil-web (results are POSTed to
                           ``<url>/callbacks/steganography`` / ``/steganalysis``).
    ``REDIS_URL``        — Celery broker + result backend.
    ``WEIGHTS_DIR``      — root of the weights/ tree (persistent volume in prod).
    ``CUDA``             — when true (and a GPU is available) run inference on CUDA.
    """

    veil_core_token: str = "dev-token"
    veil_callback_token: str = "dev-callback-token"

    web_callback_url: str = "http://localhost:3000"
    redis_url: str = "redis://localhost:6379/0"

    weights_dir: Path = Path(__file__).resolve().parent.parent / "weights"
    cuda: bool = False

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)


settings = Settings()
