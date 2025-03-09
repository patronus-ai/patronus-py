import dataclasses
import logging
from opentelemetry import trace
from typing import Optional
from typing import TYPE_CHECKING

from patronus.context.context_utils import ContextObject
from patronus.exceptions import UninitializedError

if TYPE_CHECKING:
    from patronus.evals.exporter import BatchEvaluationExporter
    from patronus.tracing.logger import Logger as PatLogger
    from patronus.api import API
    from opentelemetry.sdk.trace import TracerProvider


@dataclasses.dataclass(frozen=True)
class PatronusScope:
    project_name: Optional[str]
    app: Optional[str]
    experiment_id: Optional[str]


@dataclasses.dataclass(frozen=True)
class PatronusContext:
    scope: PatronusScope
    logger: logging.Logger
    pat_logger: "PatLogger"
    tracer_provider: "TracerProvider"
    tracer: trace.Tracer
    api_client: "API"
    exporter: "BatchEvaluationExporter"


_CTX_PAT = ContextObject[PatronusContext]("ctx.pat")


def set_global_patronus_context(ctx: PatronusContext):
    _CTX_PAT.set_global(ctx)


def get_current_context_or_none() -> Optional[PatronusContext]:
    return _CTX_PAT.get()


def get_current_context() -> PatronusContext:
    ctx = get_current_context_or_none()
    if ctx is None:
        raise UninitializedError(
            "No active Patronus context found. Please initialize the library by calling patronus.init()."
        )
    return ctx


def get_logger() -> logging.Logger:
    return get_current_context().logger


def get_logger_or_none() -> Optional[logging.Logger]:
    return (ctx := get_current_context_or_none()) and ctx.logger


def get_pat_logger_or_none() -> Optional["PatLogger"]:
    return (ctx := get_current_context_or_none()) and ctx.pat_logger


def get_tracer() -> trace.Tracer:
    return get_current_context().tracer


def get_tracer_or_none() -> Optional[trace.Tracer]:
    return (ctx := get_current_context_or_none()) and ctx.tracer


def get_api_client() -> "API":
    return get_current_context().api_client


def get_api_client_or_none() -> Optional["API"]:
    return (ctx := get_current_context_or_none()) and ctx.api_client


def get_exporter() -> "BatchEvaluationExporter":
    return get_current_context().exporter


def get_exporter_or_none() -> Optional["BatchEvaluationExporter"]:
    return (ctx := get_current_context_or_none()) and ctx.exporter


def get_scope() -> PatronusScope:
    return get_current_context().scope


def get_scope_or_none() -> Optional[PatronusScope]:
    return (ctx := get_current_context_or_none()) and ctx.scope
