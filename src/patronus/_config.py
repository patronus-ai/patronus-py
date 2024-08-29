import functools
import typing

import pydantic
import pydantic_settings


class Config(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(env_prefix="patronusai_", yaml_file="patronusai.yaml")

    api_key: str | None = pydantic.Field(None)
    api_url: str = pydantic.Field("https://api.patronus.ai")
    ui_url: str = pydantic.Field("https://app.patronus.ai")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: typing.Type[pydantic_settings.BaseSettings],
        init_settings: pydantic_settings.PydanticBaseSettingsSource,
        env_settings: pydantic_settings.PydanticBaseSettingsSource,
        dotenv_settings: pydantic_settings.PydanticBaseSettingsSource,
        file_secret_settings: pydantic_settings.PydanticBaseSettingsSource,
    ) -> typing.Tuple[pydantic_settings.PydanticBaseSettingsSource, ...]:
        return (pydantic_settings.YamlConfigSettingsSource(settings_cls),)


@functools.lru_cache()
def config() -> Config:
    cfg = Config()
    return cfg
