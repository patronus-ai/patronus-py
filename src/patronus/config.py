import platform
import os

import functools
import typing
from typing import Optional

import pydantic
import pydantic_settings

DEFAULT_API_URL = "https://api.patronus.ai"
DEFAULT_OTEL_ENDPOINT = "https://otel.patronus.ai:4317"
DEFAULT_UI_URL = "https://app.patronus.ai"

DEFAULT_PROJECT_NAME = "Global"

DEFAULT_APP_NAME = "default"


def get_service_default():
    otel_service_name = os.getenv("OTEL_SERVICE_NAME")
    if otel_service_name:
        return otel_service_name
    service = None
    try:
        service = platform.node()
    except Exception:
        pass
    if not service:
        service = "unknown_service"
    return service


class Config(pydantic_settings.BaseSettings):
    """
    Configuration settings for the Patronus SDK.

    This class defines all available configuration options with their default values
    and handles loading configuration from environment variables and YAML files.

    Configuration sources are checked in this order:

    1. Code-specified values
    2. Environment variables (with prefix PATRONUS_)
    3. YAML configuration file (patronus.yaml)
    4. Default values

    Attributes:
        service: The name of the service or application component.
            Defaults to OTEL_SERVICE_NAME env var or platform.node().
        api_key: Authentication key for Patronus services.
        api_url: URL for the Patronus API service. Default: https://api.patronus.ai
        otel_endpoint: Endpoint for OpenTelemetry data collection.
            Default: https://otel.patronus.ai:4317
        ui_url: URL for the Patronus UI. Default: https://app.patronus.ai
        timeout_s: Timeout in seconds for HTTP requests. Default: 300
        project_name: Name of the project for organizing evaluations and experiments.
            Default: Global
        app: Name of the application within the project. Default: default
    """

    model_config = pydantic_settings.SettingsConfigDict(env_prefix="patronus_", yaml_file="patronus.yaml")

    service: str = pydantic.Field(
        default_factory=get_service_default,
        description=(
            "The name of the service or application component being configured. "
            "Recommended to set the same value as `app` although not required."
            "If not provided, the `OTEL_SERVICE_NAME` environment variable will be searched. "
            "Fallbacks to `platform.node()`."
        ),
    )

    api_key: Optional[str] = pydantic.Field(default=None)
    api_url: str = pydantic.Field(default=DEFAULT_API_URL)
    otel_endpoint: str = pydantic.Field(default=DEFAULT_OTEL_ENDPOINT)
    ui_url: str = pydantic.Field(default=DEFAULT_UI_URL)

    timeout_s: int = pydantic.Field(
        default=300,
        description=(
            "Timeout for http client connecting to Patronus APIs. "
            "This includes calls for remote evaluations which may take more time "
            "than most API calls. Because of that the timeout should be greater than usual."
        ),
    )

    project_name: str = pydantic.Field(default=DEFAULT_PROJECT_NAME)
    app: str = pydantic.Field(default=DEFAULT_APP_NAME)

    @pydantic.model_validator(mode="after")
    def validate_otel_endpoint(self) -> "Config":
        if self.api_url != DEFAULT_API_URL and self.otel_endpoint == DEFAULT_OTEL_ENDPOINT:
            raise ValueError(
                "configuration error: 'api_url' is set to non-default value, "
                "but 'otel_endpoint' is a default. Change 'otel_endpoint' to point to the same environment as 'api_url'"
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: typing.Type[pydantic_settings.BaseSettings],
        init_settings: pydantic_settings.PydanticBaseSettingsSource,
        env_settings: pydantic_settings.PydanticBaseSettingsSource,
        dotenv_settings: pydantic_settings.PydanticBaseSettingsSource,
        file_secret_settings: pydantic_settings.PydanticBaseSettingsSource,
    ) -> typing.Tuple[pydantic_settings.PydanticBaseSettingsSource, ...]:
        return (
            pydantic_settings.YamlConfigSettingsSource(settings_cls),
            pydantic_settings.EnvSettingsSource(settings_cls),
        )


@functools.lru_cache()
def config() -> Config:
    """
    Returns the Patronus SDK configuration singleton.

    Configuration is loaded from environment variables and the patronus.yaml file
    (if present) when this function is first called.

    Returns:
        Config: A singleton Config object containing all Patronus configuration settings.

    Example:
        ```python
        from patronus.config import config

        # Get the configuration
        cfg = config()

        # Access configuration values
        api_key = cfg.api_key
        project_name = cfg.project_name
        ```
    """
    cfg = Config()
    return cfg
