import functools
import typing
from typing import Optional

import pydantic
import pydantic_settings

_DEFAULT_API_URL = "https://api.patronus.ai"
_DEFAULT_OTEL_ENDPOINT = "https://otel.patronus.ai:4317"


class Config(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(env_prefix="patronus_", yaml_file="patronus.yaml")

    api_key: Optional[str] = pydantic.Field(None)
    api_url: str = pydantic.Field(_DEFAULT_API_URL)
    otel_endpoint: str = pydantic.Field(_DEFAULT_OTEL_ENDPOINT)
    ui_url: str = pydantic.Field("https://app.patronus.ai")

    project_name: str = pydantic.Field(default="Global")
    app: str = pydantic.Field(default="default")

    @pydantic.model_validator(mode="after")
    def validate_otel_endpoint(self) -> "Config":
        if self.api_url != _DEFAULT_API_URL and self.otel_endpoint == _DEFAULT_OTEL_ENDPOINT:
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
