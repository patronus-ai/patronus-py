# Integrations

## patronus.integrations

This package provides integration points for connecting various third-party libraries and tools with the Patronus SDK.

### instrumenter

#### BasePatronusIntegrator

Bases: `ABC`

Abstract base class for Patronus integrations.

This class defines the interface for integrating external libraries and tools with the Patronus context. All specific integrators should inherit from this class and implement the required methods.

##### apply

```python
apply(ctx: PatronusContext, **kwargs: Any)

```

Apply the integration to the given Patronus context.

This method must be implemented by subclasses to define how the integration is applied to a Patronus context instance.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `PatronusContext` | The Patronus context to apply the integration to. | *required* | | `**kwargs` | `Any` | Additional keyword arguments specific to the implementation. | `{}` |

Source code in `src/patronus/integrations/instrumenter.py`

```python
@abc.abstractmethod
def apply(self, ctx: "context.PatronusContext", **kwargs: typing.Any):
    """
    Apply the integration to the given Patronus context.

    This method must be implemented by subclasses to define how the
    integration is applied to a Patronus context instance.

    Args:
        ctx: The Patronus context to apply the integration to.
        **kwargs: Additional keyword arguments specific to the implementation.
    """

```

### otel

#### OpenTelemetryIntegrator

```python
OpenTelemetryIntegrator(instrumentor: BaseInstrumentor)

```

Bases: `BasePatronusIntegrator`

Integration for OpenTelemetry instrumentors with Patronus.

This class provides an adapter between OpenTelemetry instrumentors and the Patronus context, allowing for easy integration of OpenTelemetry instrumentation in Patronus-managed applications.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `instrumentor` | `BaseInstrumentor` | An OpenTelemetry instrumentor instance that will be applied to the Patronus context. | *required* |

Source code in `src/patronus/integrations/otel.py`

```python
def __init__(self, instrumentor: "BaseInstrumentor"):
    """
    Initialize the OpenTelemetry integrator.

    Args:
        instrumentor: An OpenTelemetry instrumentor instance that will be
            applied to the Patronus context.
    """
    self.instrumentor = instrumentor

```

##### apply

```python
apply(ctx: PatronusContext, **kwargs: Any)

```

Apply OpenTelemetry instrumentation to the Patronus context.

This method configures the OpenTelemetry instrumentor with the tracer provider from the Patronus context.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `PatronusContext` | The Patronus context containing the tracer provider. | *required* | | `**kwargs` | `Any` | Additional keyword arguments (unused). | `{}` |

Source code in `src/patronus/integrations/otel.py`

```python
def apply(self, ctx: "context.PatronusContext", **kwargs: typing.Any):
    """
    Apply OpenTelemetry instrumentation to the Patronus context.

    This method configures the OpenTelemetry instrumentor with the
    tracer provider from the Patronus context.

    Args:
        ctx: The Patronus context containing the tracer provider.
        **kwargs: Additional keyword arguments (unused).
    """
    self.instrumentor.instrument(tracer_provider=ctx.tracer_provider)

```

### pydantic_ai

#### PydanticAIIntegrator

```python
PydanticAIIntegrator(
    event_mode: Literal["attributes", "logs"] = "logs",
)

```

Bases: `BasePatronusIntegrator`

Integration for Pydantic-AI with Patronus.

This class provides integration between Pydantic-AI agents and the Patronus observability stack, enabling tracing and logging of Pydantic-AI agent operations.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event_mode` | `Literal['attributes', 'logs']` | The mode for capturing events, either as span attributes or as logs. Default is "logs". | `'logs'` |

Source code in `src/patronus/integrations/pydantic_ai.py`

```python
def __init__(self, event_mode: Literal["attributes", "logs"] = "logs"):
    """
    Initialize the Pydantic-AI integrator.

    Args:
        event_mode: The mode for capturing events, either as span attributes
            or as logs. Default is "logs".
    """
    self._instrumentation_settings = {"event_mode": event_mode}

```

##### apply

```python
apply(ctx: PatronusContext, **kwargs: Any)

```

Apply Pydantic-AI instrumentation to the Patronus context.

This method configures all Pydantic-AI agents to use the tracer and logger providers from the Patronus context.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `PatronusContext` | The Patronus context containing the tracer and logger providers. | *required* | | `**kwargs` | `Any` | Additional keyword arguments (unused). | `{}` |

Source code in `src/patronus/integrations/pydantic_ai.py`

```python
def apply(self, ctx: "context.PatronusContext", **kwargs: Any):
    """
    Apply Pydantic-AI instrumentation to the Patronus context.

    This method configures all Pydantic-AI agents to use the tracer and logger
    providers from the Patronus context.

    Args:
        ctx: The Patronus context containing the tracer and logger providers.
        **kwargs: Additional keyword arguments (unused).
    """
    from pydantic_ai.agent import Agent, InstrumentationSettings

    settings_kwargs = {
        **self._instrumentation_settings,
        "tracer_provider": ctx.tracer_provider,
        "event_logger_provider": EventLoggerProvider(ctx.logger_provider),
    }
    settings = InstrumentationSettings(**settings_kwargs)
    Agent.instrument_all(instrument=settings)

```
