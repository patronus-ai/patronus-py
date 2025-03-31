import typing

import warnings

from typing import Optional

import httpx

from . import config
from . import context
from .api.api_client import PatronusAPIClient
from .evals.exporter import BatchEvaluationExporter
from .tracing.logger import create_logger, create_patronus_logger
from .tracing.tracer import create_tracer_provider
from .utils import Once

_INIT_ONCE = Once()


def init(
    project_name: Optional[str] = None,
    app: Optional[str] = None,
    api_url: Optional[str] = None,
    otel_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    service: Optional[str] = None,
    **kwargs: typing.Any,
) -> context.PatronusContext:
    """
    Initializes the Patronus SDK with the specified configuration.

    This function sets up the SDK with project details, API connections, and telemetry.
    It must be called before using evaluators or experiments to ensure proper recording
    of results and metrics.

    Note:
        `init()` should not be used for running experiments.
        Experiments have its own initialization process.
        You can configure them by passing configuration options to [`run_experiment()`][patronus.experiments.experiment.run_experiment]
        or using configuration file.

    Args:
        project_name: Name of the project for organizing evaluations and experiments.
            Falls back to configuration file, then defaults to "Global" if not provided.
        app: Name of the application within the project.
            Falls back to configuration file, then defaults to "default" if not provided.
        api_url: URL for the Patronus API service.
            Falls back to configuration file or environment variables if not provided.
        otel_endpoint: Endpoint for OpenTelemetry data collection.
            Falls back to configuration file or environment variables if not provided.
        api_key: Authentication key for Patronus services.
            Falls back to configuration file or environment variables if not provided.
        service: Service name for OpenTelemetry traces.
            Falls back to configuration file or environment variables if not provided.
        **kwargs: Additional configuration options for the SDK.

    Returns:
        PatronusContext: The initialized context object.

    Example:
        ```python
        import patronus

        # Load configuration from configuration file or environment variables
        patronus.init()

        # Custom initialization
        patronus.init(
            project_name="my-project",
            app="recommendation-service",
            api_key="your-api-key"
        )
        ```
    """
    if api_url != config.DEFAULT_API_URL and otel_endpoint == config.DEFAULT_OTEL_ENDPOINT:
        raise ValueError(
            "'api_url' is set to non-default value, "
            "but 'otel_endpoint' is a default. Change 'otel_endpoint' to point to the same environment as 'api_url'"
        )

    def build_and_set():
        cfg = config.config()
        ctx = build_context(
            service=service or cfg.service,
            project_name=project_name or cfg.project_name,
            app=app or cfg.app,
            experiment_id=None,
            experiment_name=None,
            api_url=api_url or cfg.api_url,
            otel_endpoint=otel_endpoint or cfg.otel_endpoint,
            api_key=api_key or cfg.api_key,
            timeout_s=cfg.timeout_s,
            **kwargs,
        )
        context.set_global_patronus_context(ctx)

    inited_now = _INIT_ONCE.do_once(build_and_set)
    if not inited_now:
        warnings.warn(
            ("The Patronus SDK has already been initialized. " "Duplicate initialization attempts are ignored."),
            UserWarning,
            stacklevel=2,
        )
    return context.get_current_context()


def build_context(
    service: str,
    project_name: str,
    app: Optional[str],
    experiment_id: Optional[str],
    experiment_name: Optional[str],
    api_url: Optional[str],
    otel_endpoint: str,
    api_key: str,
    client_http: Optional[httpx.Client] = None,
    client_http_async: Optional[httpx.AsyncClient] = None,
    timeout_s: int = 60,
    **kwargs: typing.Any,
) -> context.PatronusContext:
    """
    Builds a Patronus context with the specified configuration parameters.

    This function creates the context object that contains all necessary components
    for the SDK operation, including loggers, tracers, and API clients. It is used
    internally by the [`init()`][patronus.init.init] function but can also be used directly for more
    advanced configuration scenarios.

    Args:
        service: Service name for OpenTelemetry traces.
        project_name: Name of the project for organizing evaluations and experiments.
        app: Name of the application within the project.
        experiment_id: Unique identifier for an experiment when running in experiment mode.
        experiment_name: Display name for an experiment when running in experiment mode.
        api_url: URL for the Patronus API service.
        otel_endpoint: Endpoint for OpenTelemetry data collection.
        api_key: Authentication key for Patronus services.
        client_http: Custom HTTP client for synchronous API requests.
            If not provided, a new client will be created.
        client_http_async: Custom HTTP client for asynchronous API requests.
            If not provided, a new client will be created.
        timeout_s: Timeout in seconds for HTTP requests (default: 60).
        **kwargs: Additional configuration options, including:
            - integrations: List of OpenTelemetry instrumentors to enable.

    Returns:
        PatronusContext: The initialized context object containing all necessary
            components for SDK operation.
    """
    if client_http is None:
        client_http = httpx.Client(timeout=timeout_s)
    if client_http_async is None:
        client_http_async = httpx.AsyncClient(timeout=timeout_s)
    scope = context.PatronusScope(
        service=service,
        project_name=project_name,
        app=app,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )
    api = PatronusAPIClient(
        client_http_async=client_http_async,
        client_http=client_http,
        base_url=api_url,
        api_key=api_key,
    )
    std_logger = create_logger(
        scope=scope,
        exporter_endpoint=otel_endpoint,
        api_key=api_key,
    )
    eval_logger = create_patronus_logger(
        scope=scope,
        exporter_endpoint=otel_endpoint,
        api_key=api_key,
    )
    tracer_provider = create_tracer_provider(
        exporter_endpoint=otel_endpoint,
        api_key=api_key,
        scope=scope,
    )
    if integrations := kwargs.get("integrations"):
        if not isinstance(integrations, list):
            integrations = [integrations]

        try:
            from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

            for integration in integrations:
                if isinstance(integration, BaseInstrumentor):
                    integration.instrument(tracer_provider=tracer_provider)
                else:
                    warnings.warn(f"Integration {integration} not recognized.")
        except ImportError:
            warnings.warn("Opentelemetry instrumentation is not installed. Ignoring integrations.")

    tracer = tracer_provider.get_tracer("patronus.sdk")

    eval_exporter = BatchEvaluationExporter(client=api)
    return context.PatronusContext(
        scope=scope,
        logger=std_logger,
        pat_logger=eval_logger,
        tracer_provider=tracer_provider,
        tracer=tracer,
        api_client=api,
        exporter=eval_exporter,
    )
