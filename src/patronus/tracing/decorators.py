import functools
import inspect
import typing
from typing import Optional

from opentelemetry._logs import SeverityNumber

from .attributes import LogTypes
from .logger import Logger, get_patronus_logger
from .trace import get_tracer


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
    **kwargs,
):
    def decorator(func):
        name = span_name or func.__qualname__
        sig = inspect.signature(func)

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
            # TODO may return None for un"init"ed program
            #      Make sure it's handled gracefully
            logger = get_patronus_logger()
            # TODO may return None for un"init"ed program
            #      Make sure it's handled gracefully
            tracer = get_tracer()

            exc = None
            ret = None
            with tracer.start_as_current_span(name, record_exception=not disable_log):
                try:
                    ret = func(*f_args, **f_kwargs)
                except Exception as e:
                    exc = e
                    raise exc
                finally:
                    log_call(logger, f_args, f_kwargs, ret, exc)

                return ret

        @functools.wraps(func)
        async def wrapper_async(*f_args, **f_kwargs):
            # TODO may return None for un"init"ed program
            #      Make sure it's handled gracefully
            logger = get_patronus_logger()
            # TODO may return None for un"init"ed program
            #      Make sure it's handled gracefully
            tracer = get_tracer()

            exc = None
            ret = None
            with tracer.start_as_current_span(name, record_exception=not disable_log):
                try:
                    ret = await func(*f_args, **f_kwargs)
                except Exception as e:
                    exc = e
                    raise exc
                finally:
                    log_call(logger, f_args, f_kwargs, ret, exc)

                return ret

        if inspect.iscoroutinefunction(func):
            return wrapper_async
        else:
            return wrapper_sync

    return decorator
