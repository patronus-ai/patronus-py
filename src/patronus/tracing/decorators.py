import contextlib
import functools
import inspect
import typing
from typing import Optional

from opentelemetry.util.types import Attributes
from opentelemetry._logs import SeverityNumber

from patronus.tracing.attributes import LogTypes
from patronus.tracing.logger import Logger
from patronus import context


@contextlib.contextmanager
def start_span(name: str, *, record_exception: bool = True, attributes: Optional[Attributes] = None):
    ctx = context.get_current_context_or_none()
    if ctx is None:
        yield
        return
    with ctx.tracer.start_as_current_span(
        name,
        record_exception=record_exception,
        attributes=attributes,
    ) as span:
        yield span


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
    **kwargs,
):
    def decorator(func):
        name = span_name or func.__qualname__
        sig = inspect.signature(func)
        record_exception = not disable_log and log_exceptions

        def log_call(logger: Logger, fn_args: typing.Any, fn_kwargs: typing.Any, ret: typing.Any, exc: Exception):
            if disable_log:
                return

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
            ctx = context.get_current_context_or_none()
            if ctx is None:
                return func(*f_args, **f_kwargs)

            exc = None
            ret = None
            with ctx.tracer.start_as_current_span(name, record_exception=record_exception, attributes=attributes):
                try:
                    ret = func(*f_args, **f_kwargs)
                except Exception as e:
                    exc = e
                    raise exc
                finally:
                    log_call(ctx.pat_logger, f_args, f_kwargs, ret, exc)

                return ret

        @functools.wraps(func)
        async def wrapper_async(*f_args, **f_kwargs):
            ctx = context.get_current_context_or_none()
            if ctx is None:
                return await func(*f_args, **f_kwargs)

            exc = None
            ret = None
            with ctx.tracer.start_as_current_span(name, record_exception=record_exception, attributes=attributes):
                try:
                    ret = await func(*f_args, **f_kwargs)
                except Exception as e:
                    exc = e
                    raise exc
                finally:
                    log_call(ctx.pat_logger, f_args, f_kwargs, ret, exc)

                return ret

        if inspect.iscoroutinefunction(func):
            wrapper_async._pat_traced = True
            return wrapper_async
        else:
            wrapper_async._pat_traced = True
            return wrapper_sync

    return decorator
