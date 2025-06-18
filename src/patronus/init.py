import pathlib
import typing

import warnings

from typing import Optional

import httpx
import patronus_api

from . import config
from . import context
from .api.api_client import PatronusAPIClient
from .evals.exporter import BatchEvaluationExporter
from .integrations.instrumenter import BasePatronusIntegrator
from .integrations.otel import OpenTelemetryIntegrator
from .tracing.logger import create_logger_provider
from .tracing.tracer import create_tracer_provider
from .utils import Once

_INIT_ONCE = Once()


def init(
    project_name: Optional[str] = None,
    app: Optional[str] = None,
    api_url: Optional[str] = None,
    otel_endpoint: Optional[str] = None,
    otel_exporter_otlp_protocol: Optional[str] = None,
    api_key: Optional[str] = None,
    service: Optional[str] = None,
    resource_dir: Optional[str] = None,
    prompt_providers: Optional[list[str]] = None,
    prompt_templating_engine: Optional[str] = None,
    integrations: Optional[list[typing.Any]] = None,
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
        otel_exporter_otlp_protocol: OpenTelemetry exporter protocol (grpc or http/protobuf).
            Falls back to configuration file or environment variables if not provided.
        api_key: Authentication key for Patronus services.
            Falls back to configuration file or environment variables if not provided.
        service: Service name for OpenTelemetry traces.
            Falls back to configuration file or environment variables if not provided.
        integrations: List of integration to use.
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
            otel_exporter_otlp_protocol=otel_exporter_otlp_protocol or cfg.otel_exporter_otlp_protocol,
            api_key=api_key or cfg.api_key,
            resource_dir=resource_dir or cfg.resource_dir,
            prompt_providers=prompt_providers or cfg.prompt_providers,
            prompt_templating_engine=cfg.prompt_templating_engine,
            timeout_s=cfg.timeout_s,
            integrations=integrations,
            **kwargs,
        )
        context.set_global_patronus_context(ctx)

    inited_now = _INIT_ONCE.do_once(build_and_set)
    if not inited_now:
        warnings.warn(
            ("The Patronus SDK has already been initialized. Duplicate initialization attempts are ignored."),
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
    otel_exporter_otlp_protocol: Optional[str],
    api_key: str,
    resource_dir: Optional[str] = None,
    prompt_providers: Optional[list[str]] = None,
    prompt_templating_engine: Optional[str] = None,
    client_http: Optional[httpx.Client] = None,
    client_http_async: Optional[httpx.AsyncClient] = None,
    timeout_s: int = 60,
    integrations: Optional[list[typing.Any]] = None,
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
        otel_exporter_otlp_protocol: OpenTelemetry exporter protocol (grpc or http/protobuf).
        api_key: Authentication key for Patronus services.
        client_http: Custom HTTP client for synchronous API requests.
            If not provided, a new client will be created.
        client_http_async: Custom HTTP client for asynchronous API requests.
            If not provided, a new client will be created.
        timeout_s: Timeout in seconds for HTTP requests (default: 60).
        integrations: List of PatronusIntegrator instances.
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

    integrations = prepare_integrations(integrations)

    scope = context.PatronusScope(
        service=service,
        project_name=project_name,
        app=app,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )
    api_deprecated = PatronusAPIClient(
        client_http_async=client_http_async,
        client_http=client_http,
        base_url=api_url,
        api_key=api_key,
    )
    api_client = patronus_api.Client(api_key=api_key, base_url=api_url)
    async_api_client = patronus_api.AsyncClient(api_key=api_key, base_url=api_url)

    logger_provider = create_logger_provider(
        exporter_endpoint=otel_endpoint,
        api_key=api_key,
        scope=scope,
        protocol=otel_exporter_otlp_protocol,
    )

    tracer_provider = create_tracer_provider(
        exporter_endpoint=otel_endpoint,
        api_key=api_key,
        scope=scope,
        protocol=otel_exporter_otlp_protocol,
    )

    eval_exporter = BatchEvaluationExporter(client=api_deprecated)
    ctx = context.PatronusContext(
        scope=scope,
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        api_client_deprecated=api_deprecated,
        api_client=api_client,
        async_api_client=async_api_client,
        exporter=eval_exporter,
        prompts=context.PromptsConfig(
            directory=resource_dir and pathlib.Path(resource_dir, "prompts"),
            providers=prompt_providers,
            templating_engine=prompt_templating_engine,
        ),
    )
    apply_integrations(ctx, integrations)
    return ctx


def apply_integrations(ctx: context.PatronusContext, integrations: list[BasePatronusIntegrator]):
    for integration in integrations:
        integration.apply(ctx=ctx)


def prepare_integrations(integrations: Optional[list[typing.Any]]) -> list[BasePatronusIntegrator]:
    if not integrations:
        return []

    def map_integration(integration: typing.Any):
        if isinstance(integration, BasePatronusIntegrator):
            return integration
        try:
            from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

            if isinstance(integration, BaseInstrumentor):
                return OpenTelemetryIntegrator(integration)
        except ImportError:
            pass
        raise ValueError(f"Unrecognized integration type: {integration!r}.")

    return [map_integration(intg) for intg in integrations]
