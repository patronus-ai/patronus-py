from typing import Optional
from patronus.tracing.trace import init_tracer
from patronus.tracing.logger import init_logger
from patronus.config import config

_GLOBAL_LOGGER = None
_GLOBAL_TRACER = None

def get_tracer():
    if _GLOBAL_TRACER:
        return _GLOBAL_TRACER
    return init_tracer()

def get_logger(project_name: str):
    if _GLOBAL_LOGGER:
        return _GLOBAL_LOGGER
    return init_logger(project_name)

def _set_global_logger(_logger):
    global _GLOBAL_LOGGER
    _GLOBAL_LOGGER = _logger

def _set_global_tracer(_tracer):
    global _GLOBAL_TRACER
    _GLOBAL_TRACER = _tracer

def init(
    project_name: Optional[str],
):
    cfg = config()
    project_name = project_name or cfg.project_name

    tracer = init_tracer()
    _set_global_tracer(tracer)

    logger = init_logger(project_name)
    _set_global_logger(logger)