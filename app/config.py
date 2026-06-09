from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent


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

    Loading order (highest priority first):
      1. real environment variables — set by ``docker compose`` (env_file +
         per-service ``environment:``) and by Kamal in production;
      2. the dev ``env_file`` files below — so a bare ``uvicorn``/``celery`` run
         (no Docker) still picks up the dev tokens + localhost wiring;
      3. the defaults here.
    The dev env files have localhost defaults; Docker overrides the hostnames.
    """

    veil_core_token: str = "dev-token"
    veil_callback_token: str = "dev-callback-token"

    web_callback_url: str = "http://localhost:3000"
    redis_url: str = "redis://localhost:6379/0"

    weights_dir: Path = _ROOT / "weights"
    cuda: bool = False

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
        # Missing files are silently skipped (e.g. in the prod image).
        env_file=(
            _ROOT / "docker/secret-envs/veil-core.env",
            _ROOT / "docker/envs/core-dev.env",
        ),
        env_file_encoding="utf-8",
    )


settings = Settings()
