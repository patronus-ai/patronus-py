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


class Config(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(env_prefix="patronus_", yaml_file="patronus.yaml")

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
    cfg = Config()
    return cfg
