from typing import Optional

import httpx

from . import config
from .api import API, CTX_API
from .evals.exporter import _CTX_EVAL_EXPORTER, BatchEvaluationExporter
from .tracing.logger import _CTX_EVAL_LOGGER, _CTX_STD_LOGGER, create_logger, create_patronus_logger
from .tracing.trace import _CTX_TRACER, create_tracer


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

    project_name = project_name or cfg.project_name
    app = app or cfg.app
    api_url = api_url or cfg.api_url
    otel_endpoint = otel_endpoint or cfg.otel_endpoint
    api_key = api_key or cfg.api_key

    api = API(
        http=httpx.AsyncClient(),
        http_sync=httpx.Client(),
        base_url=api_url,
        api_key=api_key,
    )
    CTX_API.set_global(api)

    std_logger = create_logger(
        project_name=project_name,
        app=app,
        experiment_id=None,
        exporter_endpoint=otel_endpoint,
        api_key=api_key,
    )
    eval_logger = create_patronus_logger(
        project_name=project_name,
        app=app,
        experiment_id=None,
        exporter_endpoint=otel_endpoint,
        api_key=api_key,
    )
    _CTX_STD_LOGGER.set_global(std_logger)
    _CTX_EVAL_LOGGER.set_global(eval_logger)

    tracer = create_tracer(
        project_name=project_name,
        app=app,
        experiment_id=None,
        exporter_endpoint=otel_endpoint,
        api_key=api_key,
    )
    _CTX_TRACER.set_global(tracer)

    eval_exporter = BatchEvaluationExporter(client=api)
    _CTX_EVAL_EXPORTER.set_global(eval_exporter)
