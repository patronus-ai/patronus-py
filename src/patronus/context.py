import dataclasses
import logging
from opentelemetry import trace
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from patronus.tracing.logger import Logger as PatLogger
    from patronus.api import API


@dataclasses.dataclass
class PatronusScope:
    project_name: Optional[str]
    app: Optional[str]
    experiment_id: Optional[str]


@dataclasses.dataclass
class PatronusContext:
    scope: PatronusScope
    logger: logging.Logger
    pat_logger: "PatLogger"
    tracer: trace.Tracer
    api_client: "API"
