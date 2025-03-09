from typing import Optional

import httpx

from . import config
from . import context
from .api import API
from .evals.exporter import BatchEvaluationExporter
from .tracing.logger import create_logger, create_patronus_logger
from .tracing.trace import create_tracer


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
):
    if api_url != config._DEFAULT_API_URL and otel_endpoint == config._DEFAULT_OTEL_ENDPOINT:
        raise ValueError(
            "'api_url' is set to non-default value, "
            "but 'otel_endpoint' is a default. Change 'otel_endpoint' to point to the same environment as 'api_url'"
        )

    cfg = config.config()
    ctx = build_context(
        project_name=project_name or cfg.project_name,
        app=app or cfg.app,
        api_url=api_url or cfg.api_url,
        otel_endpoint=otel_endpoint or cfg.otel_endpoint,
        api_key=api_key or cfg.api_key,
    )
    context.set_global_patronus_context(ctx)


def build_context(
    project_name: str,
    app: Optional[str],
    api_url: Optional[str],
    otel_endpoint: str,
    api_key: str,
) -> context.PatronusContext:
    scope = context.PatronusScope(
        project_name=project_name,
        app=app,
        experiment_id=None,
    )
    api = API(
        http=httpx.AsyncClient(),
        http_sync=httpx.Client(),
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
    tracer = create_tracer(
        exporter_endpoint=otel_endpoint,
        api_key=api_key,
        scope=scope,
    )
    eval_exporter = BatchEvaluationExporter(client=api)
    return context.PatronusContext(
        scope=scope,
        logger=std_logger,
        pat_logger=eval_logger,
        tracer=tracer,
        api_client=api,
        exporter=eval_exporter,
    )
