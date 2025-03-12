import warnings

from typing import Optional

import httpx

from . import config
from . import context
from .api.api_client import PatronusAPIClient
from .evals.exporter import BatchEvaluationExporter
from .tracing.logger import create_logger, create_patronus_logger
from .tracing.tracer import create_tracer_provider


def init(
    # Initialize SDK with project name.
    # If value is not provided, then it's loaded from the configuration file.
    # Defaults to "Global" if neither is provided.
    project_name: Optional[str] = None,
    # Initialize SDK with an app.
    # If value is not provided, then it's loaded from the configuration file.
    # Defaults to "default" if neither is provided.
    app: Optional[str] = None,
    api_url: Optional[str] = None,
    otel_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
):
    if api_url != config.DEFAULT_API_URL and otel_endpoint == config.DEFAULT_OTEL_ENDPOINT:
        raise ValueError(
            "'api_url' is set to non-default value, "
            "but 'otel_endpoint' is a default. Change 'otel_endpoint' to point to the same environment as 'api_url'"
        )

    cfg = config.config()
    ctx = build_context(
        project_name=project_name or cfg.project_name,
        app=app or cfg.app,
        experiment_id=None,
        api_url=api_url or cfg.api_url,
        otel_endpoint=otel_endpoint or cfg.otel_endpoint,
        api_key=api_key or cfg.api_key,
        timeout_s=cfg.timeout_s,
        **kwargs,
    )
    context.set_global_patronus_context(ctx)


def build_context(
    project_name: str,
    app: Optional[str],
    experiment_id: Optional[str],
    api_url: Optional[str],
    otel_endpoint: str,
    api_key: str,
    client_http: Optional[httpx.Client] = None,
    client_http_async: Optional[httpx.AsyncClient] = None,
    timeout_s: int = 60,
    **kwargs,
) -> context.PatronusContext:
    if client_http is None:
        client_http = httpx.Client(timeout=timeout_s)
    if client_http_async is None:
        client_http_async = httpx.AsyncClient(timeout=timeout_s)
    scope = context.PatronusScope(
        project_name=project_name,
        app=app,
        experiment_id=experiment_id,
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
                    print(f"Instrumentation {integration} ENABLED")
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
