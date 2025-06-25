# Config

## patronus.config

### Config

Bases: `BaseSettings`

Configuration settings for the Patronus SDK.

This class defines all available configuration options with their default values and handles loading configuration from environment variables and YAML files.

Configuration sources are checked in this order:

1. Code-specified values
1. Environment variables (with prefix PATRONUS\_)
1. YAML configuration file (patronus.yaml)
1. Default values

Attributes:

| Name | Type | Description | | --- | --- | --- | | `service` | `str` | The name of the service or application component. Defaults to OTEL_SERVICE_NAME env var or platform.node(). | | `api_key` | `Optional[str]` | Authentication key for Patronus services. | | `api_url` | `str` | URL for the Patronus API service. Default: https://api.patronus.ai | | `otel_endpoint` | `str` | Endpoint for OpenTelemetry data collection. Default: https://otel.patronus.ai:4317 | | `otel_exporter_otlp_protocol` | `Optional[Literal['grpc', 'http/protobuf']]` | OpenTelemetry exporter protocol. Values: grpc, http/protobuf. Falls back to standard OTEL environment variables if not set. | | `ui_url` | `str` | URL for the Patronus UI. Default: https://app.patronus.ai | | `timeout_s` | `int` | Timeout in seconds for HTTP requests. Default: 300 | | `project_name` | `str` | Name of the project for organizing evaluations and experiments. Default: Global | | `app` | `str` | Name of the application within the project. Default: default |

### config

```python
config() -> Config

```

Returns the Patronus SDK configuration singleton.

Configuration is loaded from environment variables and the patronus.yaml file (if present) when this function is first called.

Returns:

| Name | Type | Description | | --- | --- | --- | | `Config` | `Config` | A singleton Config object containing all Patronus configuration settings. |

Example

```python
from patronus.config import config

# Get the configuration
cfg = config()

# Access configuration values
api_key = cfg.api_key
project_name = cfg.project_name

```

Source code in `src/patronus/config.py`

````python
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

````
