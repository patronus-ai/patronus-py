# Context

## patronus.context

Context management for Patronus SDK.

This module provides classes and utility functions for managing the global Patronus context and accessing different components of the SDK like logging, tracing, and API clients.

### PatronusScope

```python
PatronusScope(
    service: Optional[str],
    project_name: Optional[str],
    app: Optional[str],
    experiment_id: Optional[str],
    experiment_name: Optional[str],
)

```

Scope information for Patronus context.

Defines the scope of the current Patronus application or experiment.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `service` | `Optional[str]` | The service name as defined in OTeL. | | `project_name` | `Optional[str]` | The project name. | | `app` | `Optional[str]` | The application name. | | `experiment_id` | `Optional[str]` | The unique identifier for the experiment. | | `experiment_name` | `Optional[str]` | The name of the experiment. |

### PromptsConfig

```python
PromptsConfig(
    directory: Path,
    providers: list[str],
    templating_engine: str,
)

```

#### directory

```python
directory: Path

```

The absolute path to a directory where prompts are stored locally.

#### providers

```python
providers: list[str]

```

List of default prompt providers.

#### templating_engine

```python
templating_engine: str

```

Default prompt templating engine.

### PatronusContext

```python
PatronusContext(
    scope: PatronusScope,
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    api_client_deprecated: PatronusAPIClient,
    api_client: Client,
    async_api_client: AsyncClient,
    exporter: BatchEvaluationExporter,
    prompts: PromptsConfig,
)

```

Context object for Patronus SDK.

Contains all the necessary components for the SDK to function properly.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `scope` | `PatronusScope` | Scope information for this context. | | `tracer_provider` | `TracerProvider` | The OpenTelemetry tracer provider. | | `logger_provider` | `LoggerProvider` | The OpenTelemetry logger provider. | | `api_client_deprecated` | `PatronusAPIClient` | Client for Patronus API communication (deprecated). | | `api_client` | `Client` | Client for Patronus API communication using the modern client. | | `async_api_client` | `AsyncClient` | Asynchronous client for Patronus API communication. | | `exporter` | `BatchEvaluationExporter` | Exporter for batch evaluation results. | | `prompts` | `PromptsConfig` | Configuration for prompt management. |

### set_global_patronus_context

```python
set_global_patronus_context(ctx: PatronusContext)

```

Set the global Patronus context.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `PatronusContext` | The Patronus context to set globally. | *required* |

Source code in `src/patronus/context/__init__.py`

```python
def set_global_patronus_context(ctx: PatronusContext):
    """
    Set the global Patronus context.

    Args:
        ctx: The Patronus context to set globally.
    """
    _CTX_PAT.set_global(ctx)

```

### get_current_context_or_none

```python
get_current_context_or_none() -> Optional[PatronusContext]

```

Get the current Patronus context or None if not initialized.

Returns:

| Type | Description | | --- | --- | | `Optional[PatronusContext]` | The current PatronusContext if set, otherwise None. |

Source code in `src/patronus/context/__init__.py`

```python
def get_current_context_or_none() -> Optional[PatronusContext]:
    """
    Get the current Patronus context or None if not initialized.

    Returns:
        The current PatronusContext if set, otherwise None.
    """
    return _CTX_PAT.get()

```

### get_current_context

```python
get_current_context() -> PatronusContext

```

Get the current Patronus context.

Returns:

| Type | Description | | --- | --- | | `PatronusContext` | The current PatronusContext. |

Raises:

| Type | Description | | --- | --- | | `UninitializedError` | If no active Patronus context is found. |

Source code in `src/patronus/context/__init__.py`

```python
def get_current_context() -> PatronusContext:
    """
    Get the current Patronus context.

    Returns:
        The current PatronusContext.

    Raises:
        UninitializedError: If no active Patronus context is found.
    """
    ctx = get_current_context_or_none()
    if ctx is None:
        raise UninitializedError(
            "No active Patronus context found. Please initialize the library by calling patronus.init()."
        )
    return ctx

```

### get_logger

```python
get_logger(
    ctx: Optional[PatronusContext] = None,
    level: int = logging.INFO,
) -> logging.Logger

```

Get a standard Python logger configured with the Patronus context.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `Optional[PatronusContext]` | The Patronus context to use. If None, uses the current context. | `None` | | `level` | `int` | The logging level to set. Defaults to INFO. | `INFO` |

Returns:

| Type | Description | | --- | --- | | `Logger` | A configured Python logger. |

Source code in `src/patronus/context/__init__.py`

```python
def get_logger(ctx: Optional[PatronusContext] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get a standard Python logger configured with the Patronus context.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.
        level: The logging level to set. Defaults to INFO.

    Returns:
        A configured Python logger.
    """
    from patronus.tracing.logger import set_logger_handler

    ctx = ctx or get_current_context()

    logger = logging.getLogger("patronus.sdk")
    set_logger_handler(logger, ctx.scope, ctx.logger_provider)
    logger.setLevel(level)
    return logger

```

### get_logger_or_none

```python
get_logger_or_none(
    level: int = logging.INFO,
) -> Optional[logging.Logger]

```

Get a standard Python logger or None if context is not initialized.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `level` | `int` | The logging level to set. Defaults to INFO. | `INFO` |

Returns:

| Type | Description | | --- | --- | | `Optional[Logger]` | A configured Python logger if context is available, otherwise None. |

Source code in `src/patronus/context/__init__.py`

```python
def get_logger_or_none(level: int = logging.INFO) -> Optional[logging.Logger]:
    """
    Get a standard Python logger or None if context is not initialized.

    Args:
        level: The logging level to set. Defaults to INFO.

    Returns:
        A configured Python logger if context is available, otherwise None.
    """
    ctx = get_current_context()
    if ctx is None:
        return None
    return get_logger(ctx, level=level)

```

### get_pat_logger

```python
get_pat_logger(
    ctx: Optional[PatronusContext] = None,
) -> PatLogger

```

Get a Patronus logger.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `Optional[PatronusContext]` | The Patronus context to use. If None, uses the current context. | `None` |

Returns:

| Type | Description | | --- | --- | | `Logger` | A Patronus logger. |

Source code in `src/patronus/context/__init__.py`

```python
def get_pat_logger(ctx: Optional[PatronusContext] = None) -> "PatLogger":
    """
    Get a Patronus logger.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        A Patronus logger.
    """
    ctx = ctx or get_current_context()
    return ctx.logger_provider.get_logger("patronus.sdk")

```

### get_pat_logger_or_none

```python
get_pat_logger_or_none() -> Optional[PatLogger]

```

Get a Patronus logger or None if context is not initialized.

Returns:

| Type | Description | | --- | --- | | `Optional[Logger]` | A Patronus logger if context is available, otherwise None. |

Source code in `src/patronus/context/__init__.py`

```python
def get_pat_logger_or_none() -> Optional["PatLogger"]:
    """
    Get a Patronus logger or None if context is not initialized.

    Returns:
        A Patronus logger if context is available, otherwise None.
    """
    ctx = get_current_context_or_none()
    if ctx is None:
        return None

    return ctx.logger_provider.get_logger("patronus.sdk")

```

### get_tracer

```python
get_tracer(
    ctx: Optional[PatronusContext] = None,
) -> trace.Tracer

```

Get an OpenTelemetry tracer.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `Optional[PatronusContext]` | The Patronus context to use. If None, uses the current context. | `None` |

Returns:

| Type | Description | | --- | --- | | `Tracer` | An OpenTelemetry tracer. |

Source code in `src/patronus/context/__init__.py`

```python
def get_tracer(ctx: Optional[PatronusContext] = None) -> trace.Tracer:
    """
    Get an OpenTelemetry tracer.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        An OpenTelemetry tracer.
    """
    ctx = ctx or get_current_context()
    return ctx.tracer_provider.get_tracer("patronus.sdk")

```

### get_tracer_or_none

```python
get_tracer_or_none() -> Optional[trace.Tracer]

```

Get an OpenTelemetry tracer or None if context is not initialized.

Returns:

| Type | Description | | --- | --- | | `Optional[Tracer]` | An OpenTelemetry tracer if context is available, otherwise None. |

Source code in `src/patronus/context/__init__.py`

```python
def get_tracer_or_none() -> Optional[trace.Tracer]:
    """
    Get an OpenTelemetry tracer or None if context is not initialized.

    Returns:
        An OpenTelemetry tracer if context is available, otherwise None.
    """
    ctx = get_current_context_or_none()
    if ctx is None:
        return None
    return ctx.tracer_provider.get_tracer("patronus.sdk")

```

### get_api_client_deprecated

```python
get_api_client_deprecated(
    ctx: Optional[PatronusContext] = None,
) -> PatronusAPIClient

```

Get the Patronus API client.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `Optional[PatronusContext]` | The Patronus context to use. If None, uses the current context. | `None` |

Returns:

| Type | Description | | --- | --- | | `PatronusAPIClient` | The Patronus API client. |

Source code in `src/patronus/context/__init__.py`

```python
def get_api_client_deprecated(ctx: Optional[PatronusContext] = None) -> "PatronusAPIClient":
    """
    Get the Patronus API client.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The Patronus API client.
    """
    ctx = ctx or get_current_context()
    return ctx.api_client_deprecated

```

### get_api_client_deprecated_or_none

```python
get_api_client_deprecated_or_none() -> Optional[
    PatronusAPIClient
]

```

Get the Patronus API client or None if context is not initialized.

Returns:

| Type | Description | | --- | --- | | `Optional[PatronusAPIClient]` | The Patronus API client if context is available, otherwise None. |

Source code in `src/patronus/context/__init__.py`

```python
def get_api_client_deprecated_or_none() -> Optional["PatronusAPIClient"]:
    """
    Get the Patronus API client or None if context is not initialized.

    Returns:
        The Patronus API client if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.api_client_deprecated

```

### get_api_client

```python
get_api_client(
    ctx: Optional[PatronusContext] = None,
) -> patronus_api.Client

```

Get the Patronus API client.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `Optional[PatronusContext]` | The Patronus context to use. If None, uses the current context. | `None` |

Returns:

| Type | Description | | --- | --- | | `Client` | The Patronus API client. |

Source code in `src/patronus/context/__init__.py`

```python
def get_api_client(ctx: Optional[PatronusContext] = None) -> patronus_api.Client:
    """
    Get the Patronus API client.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The Patronus API client.
    """
    ctx = ctx or get_current_context()
    return ctx.api_client

```

### get_api_client_or_none

```python
get_api_client_or_none() -> Optional[patronus_api.Client]

```

Get the Patronus API client or None if context is not initialized.

Returns:

| Type | Description | | --- | --- | | `Optional[Client]` | The Patronus API client if context is available, otherwise None. |

Source code in `src/patronus/context/__init__.py`

```python
def get_api_client_or_none() -> Optional[patronus_api.Client]:
    """
    Get the Patronus API client or None if context is not initialized.

    Returns:
        The Patronus API client if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.api_client

```

### get_async_api_client

```python
get_async_api_client(
    ctx: Optional[PatronusContext] = None,
) -> patronus_api.AsyncClient

```

Get the asynchronous Patronus API client.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `Optional[PatronusContext]` | The Patronus context to use. If None, uses the current context. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncClient` | The asynchronous Patronus API client. |

Source code in `src/patronus/context/__init__.py`

```python
def get_async_api_client(ctx: Optional[PatronusContext] = None) -> patronus_api.AsyncClient:
    """
    Get the asynchronous Patronus API client.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The asynchronous Patronus API client.
    """
    ctx = ctx or get_current_context()
    return ctx.async_api_client

```

### get_async_api_client_or_none

```python
get_async_api_client_or_none() -> Optional[
    patronus_api.AsyncClient
]

```

Get the asynchronous Patronus API client or None if context is not initialized.

Returns:

| Type | Description | | --- | --- | | `Optional[AsyncClient]` | The asynchronous Patronus API client if context is available, otherwise None. |

Source code in `src/patronus/context/__init__.py`

```python
def get_async_api_client_or_none() -> Optional[patronus_api.AsyncClient]:
    """
    Get the asynchronous Patronus API client or None if context is not initialized.

    Returns:
        The asynchronous Patronus API client if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.async_api_client

```

### get_exporter

```python
get_exporter(
    ctx: Optional[PatronusContext] = None,
) -> BatchEvaluationExporter

```

Get the batch evaluation exporter.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `Optional[PatronusContext]` | The Patronus context to use. If None, uses the current context. | `None` |

Returns:

| Type | Description | | --- | --- | | `BatchEvaluationExporter` | The batch evaluation exporter. |

Source code in `src/patronus/context/__init__.py`

```python
def get_exporter(ctx: Optional[PatronusContext] = None) -> "BatchEvaluationExporter":
    """
    Get the batch evaluation exporter.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The batch evaluation exporter.
    """
    ctx = ctx or get_current_context()
    return ctx.exporter

```

### get_exporter_or_none

```python
get_exporter_or_none() -> Optional[BatchEvaluationExporter]

```

Get the batch evaluation exporter or None if context is not initialized.

Returns:

| Type | Description | | --- | --- | | `Optional[BatchEvaluationExporter]` | The batch evaluation exporter if context is available, otherwise None. |

Source code in `src/patronus/context/__init__.py`

```python
def get_exporter_or_none() -> Optional["BatchEvaluationExporter"]:
    """
    Get the batch evaluation exporter or None if context is not initialized.

    Returns:
        The batch evaluation exporter if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.exporter

```

### get_scope

```python
get_scope(
    ctx: Optional[PatronusContext] = None,
) -> PatronusScope

```

Get the Patronus scope.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `Optional[PatronusContext]` | The Patronus context to use. If None, uses the current context. | `None` |

Returns:

| Type | Description | | --- | --- | | `PatronusScope` | The Patronus scope. |

Source code in `src/patronus/context/__init__.py`

```python
def get_scope(ctx: Optional[PatronusContext] = None) -> PatronusScope:
    """
    Get the Patronus scope.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The Patronus scope.
    """
    ctx = ctx or get_current_context()
    return ctx.scope

```

### get_scope_or_none

```python
get_scope_or_none() -> Optional[PatronusScope]

```

Get the Patronus scope or None if context is not initialized.

Returns:

| Type | Description | | --- | --- | | `Optional[PatronusScope]` | The Patronus scope if context is available, otherwise None. |

Source code in `src/patronus/context/__init__.py`

```python
def get_scope_or_none() -> Optional[PatronusScope]:
    """
    Get the Patronus scope or None if context is not initialized.

    Returns:
        The Patronus scope if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.scope

```

### get_prompts_config

```python
get_prompts_config(
    ctx: Optional[PatronusContext] = None,
) -> PromptsConfig

```

Get the Patronus prompts configuration.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `Optional[PatronusContext]` | The Patronus context to use. If None, uses the current context. | `None` |

Returns:

| Type | Description | | --- | --- | | `PromptsConfig` | The Patronus prompts configuration. |

Source code in `src/patronus/context/__init__.py`

```python
def get_prompts_config(ctx: Optional[PatronusContext] = None) -> PromptsConfig:
    """
    Get the Patronus prompts configuration.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The Patronus prompts configuration.
    """
    ctx = ctx or get_current_context()
    return ctx.prompts

```

### get_prompts_config_or_none

```python
get_prompts_config_or_none() -> Optional[PromptsConfig]

```

Get the Patronus prompts configuration or None if context is not initialized.

Returns:

| Type | Description | | --- | --- | | `Optional[PromptsConfig]` | The Patronus prompts configuration if context is available, otherwise None. |

Source code in `src/patronus/context/__init__.py`

```python
def get_prompts_config_or_none() -> Optional[PromptsConfig]:
    """
    Get the Patronus prompts configuration or None if context is not initialized.

    Returns:
        The Patronus prompts configuration if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.prompts

```
