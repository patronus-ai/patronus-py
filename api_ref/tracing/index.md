# Tracing

## patronus.tracing

### decorators

#### start_span

```python
start_span(
    name: str,
    *,
    record_exception: bool = True,
    attributes: Optional[Attributes] = None,
) -> Iterator[Optional[typing.Any]]

```

Context manager for creating and managing a trace span.

This function is used to create a span within the current context using the tracer, allowing you to track execution timing or events within a specific block of code. The context is set by `patronus.init()` function. If SDK was not initialized, yielded value will be None.

Example:

```python
import patronus

patronus.init()

# Use context manager for finer-grained tracing
def complex_operation():
    with patronus.start_span("Data preparation"):
        # Prepare data
        pass

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str` | The name of the span. | *required* | | `record_exception` | `bool` | Whether to record exceptions that occur within the span. Default is True. | `True` | | `attributes` | `Optional[Attributes]` | Attributes to associate with the span, providing additional metadata. | `None` |

Source code in `src/patronus/tracing/decorators.py`

````python
@contextlib.contextmanager
def start_span(
    name: str, *, record_exception: bool = True, attributes: Optional[Attributes] = None
) -> Iterator[Optional[typing.Any]]:
    """
    Context manager for creating and managing a trace span.

    This function is used to create a span within the current context using the tracer,
    allowing you to track execution timing or events within a specific block of code.
    The context is set by `patronus.init()` function. If SDK was not initialized, yielded value will be None.

    Example:

    ```python
    import patronus

    patronus.init()

    # Use context manager for finer-grained tracing
    def complex_operation():
        with patronus.start_span("Data preparation"):
            # Prepare data
            pass
    ```


    Args:
        name (str): The name of the span.
        record_exception (bool): Whether to record exceptions that occur within the span. Default is True.
        attributes (Optional[Attributes]): Attributes to associate with the span, providing additional metadata.
    """
    tracer = context.get_tracer_or_none()
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(
        name,
        record_exception=record_exception,
        attributes=attributes,
    ) as span:
        yield span

````

#### traced

```python
traced(
    span_name: Optional[str] = None,
    *,
    log_args: bool = True,
    log_results: bool = True,
    log_exceptions: bool = True,
    disable_log: bool = False,
    attributes: Attributes = None,
    **kwargs: Any,
)

```

A decorator to trace function execution by recording a span for the traced function.

Example:

```python
import patronus

patronus.init()

# Trace a function with the @traced decorator
@patronus.traced()
def process_input(user_query):
    # Process the input

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `span_name` | `Optional[str]` | The name of the traced span. Defaults to the function name if not provided. | `None` | | `log_args` | `bool` | Whether to log the arguments passed to the function. Default is True. | `True` | | `log_results` | `bool` | Whether to log the function's return value. Default is True. | `True` | | `log_exceptions` | `bool` | Whether to log any exceptions raised while executing the function. Default is True. | `True` | | `disable_log` | `bool` | Whether to disable logging the trace information. Default is False. | `False` | | `attributes` | `Attributes` | Attributes to attach to the traced span. Default is None. | `None` | | `**kwargs` | `Any` | Additional arguments for the decorator. | `{}` |

Source code in `src/patronus/tracing/decorators.py`

````python
def traced(
    # Give name for the traced span. Defaults to a function name if not provided.
    span_name: Optional[str] = None,
    *,
    # Whether to log function arguments.
    log_args: bool = True,
    # Whether to log function output.
    log_results: bool = True,
    # Whether to log an exception if one was raised.
    log_exceptions: bool = True,
    # Whether to prevent a log message to be created.
    disable_log: bool = False,
    attributes: Attributes = None,
    **kwargs: typing.Any,
):
    """
    A decorator to trace function execution by recording a span for the traced function.

    Example:

    ```python
    import patronus

    patronus.init()

    # Trace a function with the @traced decorator
    @patronus.traced()
    def process_input(user_query):
        # Process the input
    ```

    Args:
        span_name (Optional[str]): The name of the traced span. Defaults to the function name if not provided.
        log_args (bool): Whether to log the arguments passed to the function. Default is True.
        log_results (bool): Whether to log the function's return value. Default is True.
        log_exceptions (bool): Whether to log any exceptions raised while executing the function. Default is True.
        disable_log (bool): Whether to disable logging the trace information. Default is False.
        attributes (Attributes): Attributes to attach to the traced span. Default is None.
        **kwargs: Additional arguments for the decorator.
    """

    def decorator(func):
        name = span_name or func.__qualname__
        sig = inspect.signature(func)
        record_exception = not disable_log and log_exceptions

        def log_call(fn_args: typing.Any, fn_kwargs: typing.Any, ret: typing.Any, exc: Exception):
            if disable_log:
                return

            logger = context.get_pat_logger()
            severity = SeverityNumber.INFO
            body = {"function.name": name}
            if log_args:
                bound_args = sig.bind(*fn_args, **fn_kwargs)
                body["function.arguments"] = {**bound_args.arguments, **bound_args.arguments}
            if log_results is not None and exc is None:
                body["function.output"] = ret
            if log_exceptions and exc is not None:
                module = type(exc).__module__
                qualname = type(exc).__qualname__
                exception_type = f"{module}.{qualname}" if module and module != "builtins" else qualname
                body["exception.type"] = exception_type
                body["exception.message"] = str(exc)
                severity = SeverityNumber.ERROR
            logger.log(body, log_type=LogTypes.trace, severity=severity)

        @functools.wraps(func)
        def wrapper_sync(*f_args, **f_kwargs):
            tracer = context.get_tracer_or_none()
            if tracer is None:
                return func(*f_args, **f_kwargs)

            exc = None
            ret = None
            with tracer.start_as_current_span(name, record_exception=record_exception, attributes=attributes):
                try:
                    ret = func(*f_args, **f_kwargs)
                except Exception as e:
                    exc = e
                    raise exc
                finally:
                    log_call(f_args, f_kwargs, ret, exc)

                return ret

        @functools.wraps(func)
        async def wrapper_async(*f_args, **f_kwargs):
            tracer = context.get_tracer_or_none()
            if tracer is None:
                return await func(*f_args, **f_kwargs)

            exc = None
            ret = None
            with tracer.start_as_current_span(name, record_exception=record_exception, attributes=attributes):
                try:
                    ret = await func(*f_args, **f_kwargs)
                except Exception as e:
                    exc = e
                    raise exc
                finally:
                    log_call(f_args, f_kwargs, ret, exc)

                return ret

        if inspect.iscoroutinefunction(func):
            wrapper_async._pat_traced = True
            return wrapper_async
        else:
            wrapper_async._pat_traced = True
            return wrapper_sync

    return decorator

````

### exporters

This module provides exporter selection functionality for OpenTelemetry traces and logs. It handles protocol resolution based on Patronus configuration and standard OTEL environment variables.

#### create_trace_exporter

```python
create_trace_exporter(
    endpoint: str,
    api_key: str,
    protocol: Optional[str] = None,
) -> SpanExporter

```

Create a configured trace exporter instance.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `endpoint` | `str` | The OTLP endpoint URL | *required* | | `api_key` | `str` | Authentication key for Patronus services | *required* | | `protocol` | `Optional[str]` | OTLP protocol override from Patronus configuration | `None` |

Returns:

| Type | Description | | --- | --- | | `SpanExporter` | Configured trace exporter instance |

Source code in `src/patronus/tracing/exporters.py`

```python
def create_trace_exporter(endpoint: str, api_key: str, protocol: Optional[str] = None) -> SpanExporter:
    """
    Create a configured trace exporter instance.

    Args:
        endpoint: The OTLP endpoint URL
        api_key: Authentication key for Patronus services
        protocol: OTLP protocol override from Patronus configuration

    Returns:
        Configured trace exporter instance
    """
    resolved_protocol = _resolve_otlp_protocol(protocol)

    if resolved_protocol == "http/protobuf":
        # For HTTP exporter, ensure endpoint has the correct path
        if not endpoint.endswith("/v1/traces"):
            endpoint = endpoint.rstrip("/") + "/v1/traces"
        return OTLPSpanExporterHTTP(endpoint=endpoint, headers={"x-api-key": api_key})
    else:
        # For gRPC exporter, determine if connection should be insecure based on URL scheme
        is_insecure = endpoint.startswith("http://")
        return OTLPSpanExporterGRPC(endpoint=endpoint, headers={"x-api-key": api_key}, insecure=is_insecure)

```

#### create_log_exporter

```python
create_log_exporter(
    endpoint: str,
    api_key: str,
    protocol: Optional[str] = None,
) -> LogExporter

```

Create a configured log exporter instance.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `endpoint` | `str` | The OTLP endpoint URL | *required* | | `api_key` | `str` | Authentication key for Patronus services | *required* | | `protocol` | `Optional[str]` | OTLP protocol override from Patronus configuration | `None` |

Returns:

| Type | Description | | --- | --- | | `LogExporter` | Configured log exporter instance |

Source code in `src/patronus/tracing/exporters.py`

```python
def create_log_exporter(endpoint: str, api_key: str, protocol: Optional[str] = None) -> LogExporter:
    """
    Create a configured log exporter instance.

    Args:
        endpoint: The OTLP endpoint URL
        api_key: Authentication key for Patronus services
        protocol: OTLP protocol override from Patronus configuration

    Returns:
        Configured log exporter instance
    """
    resolved_protocol = _resolve_otlp_protocol(protocol)

    if resolved_protocol == "http/protobuf":
        # For HTTP exporter, ensure endpoint has the correct path
        if not endpoint.endswith("/v1/logs"):
            endpoint = endpoint.rstrip("/") + "/v1/logs"
        return OTLPLogExporterHTTP(endpoint=endpoint, headers={"x-api-key": api_key})
    else:
        # For gRPC exporter, determine if connection should be insecure based on URL scheme
        is_insecure = endpoint.startswith("http://")
        return OTLPLogExporterGRPC(endpoint=endpoint, headers={"x-api-key": api_key}, insecure=is_insecure)

```

### tracer

This module provides the implementation for tracing support using the OpenTelemetry SDK.

#### PatronusAttributesSpanProcessor

```python
PatronusAttributesSpanProcessor(
    project_name: str,
    app: Optional[str] = None,
    experiment_id: Optional[str] = None,
)

```

Bases: `SpanProcessor`

Processor that adds Patronus-specific attributes to all spans.

This processor ensures that each span includes the mandatory attributes: `project_name`, and optionally adds `app` or `experiment_id` attributes if they are provided during initialization.

Source code in `src/patronus/tracing/tracer.py`

```python
def __init__(self, project_name: str, app: Optional[str] = None, experiment_id: Optional[str] = None):
    self.project_name = project_name
    self.experiment_id = None
    self.app = None

    if experiment_id is not None:
        self.experiment_id = experiment_id
    else:
        self.app = app

```

#### create_tracer_provider

```python
create_tracer_provider(
    exporter_endpoint: str,
    api_key: str,
    scope: PatronusScope,
    protocol: Optional[str] = None,
) -> TracerProvider

```

Creates and returns a cached TracerProvider configured with the specified exporter.

The function utilizes an OpenTelemetry BatchSpanProcessor and an OTLPSpanExporter to initialize the tracer. The configuration is cached for reuse.

Source code in `src/patronus/tracing/tracer.py`

```python
@functools.lru_cache()
def create_tracer_provider(
    exporter_endpoint: str,
    api_key: str,
    scope: context.PatronusScope,
    protocol: Optional[str] = None,
) -> TracerProvider:
    """
    Creates and returns a cached TracerProvider configured with the specified exporter.

    The function utilizes an OpenTelemetry BatchSpanProcessor and an
    OTLPSpanExporter to initialize the tracer. The configuration is cached for reuse.
    """
    resource = None
    if scope.service is not None:
        resource = Resource.create({"service.name": scope.service})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        PatronusAttributesSpanProcessor(
            project_name=scope.project_name,
            app=scope.app,
            experiment_id=scope.experiment_id,
        )
    )
    provider.add_span_processor(
        BatchSpanProcessor(_create_exporter(endpoint=exporter_endpoint, api_key=api_key, protocol=protocol))
    )
    return provider

```

#### create_tracer

```python
create_tracer(
    scope: PatronusScope,
    exporter_endpoint: str,
    api_key: str,
    protocol: Optional[str] = None,
) -> trace.Tracer

```

Creates an OpenTelemetry (OTeL) tracer tied to the specified scope.

Source code in `src/patronus/tracing/tracer.py`

```python
def create_tracer(
    scope: context.PatronusScope,
    exporter_endpoint: str,
    api_key: str,
    protocol: Optional[str] = None,
) -> trace.Tracer:
    """
    Creates an OpenTelemetry (OTeL) tracer tied to the specified scope.
    """
    provider = create_tracer_provider(
        exporter_endpoint=exporter_endpoint,
        api_key=api_key,
        scope=scope,
        protocol=protocol,
    )
    return provider.get_tracer("patronus.sdk")

```
