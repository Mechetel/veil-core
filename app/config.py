from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from the environment.

    ``VEIL_CORE_TOKEN`` is the shared secret the web app (veil-web) sends in the
    ``X-Auth-Token`` header on every request to the core API.
    """

    veil_core_token: str = "dev-token"

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)


settings = Settings()
