import contextlib
import functools
import inspect
import typing
from typing import Optional, Iterator

from opentelemetry.util.types import Attributes
from opentelemetry._logs import SeverityNumber

from patronus.tracing.attributes import LogTypes
from patronus import context


@contextlib.contextmanager
def start_span(
    name: str, *, record_exception: bool = True, attributes: Optional[Attributes] = None
) -> Iterator[Optional[typing.Any]]:
    """
    Context manager for creating and managing a trace span.

    This function is used to create a span within the current context using the tracer,
    allowing you to track execution timing or events within a specific block of code.
    The context is set by `patronus.init()` function. If SDK was not initialized, yielded value will be None.

    Example:

    ```python
    import patronus

    patronus.init()

    # Use context manager for finer-grained tracing
    def complex_operation():
        with patronus.start_span("Data preparation"):
            # Prepare data
            pass
    ```


    Args:
        name (str): The name of the span.
        record_exception (bool): Whether to record exceptions that occur within the span. Default is True.
        attributes (Optional[Attributes]): Attributes to associate with the span, providing additional metadata.
    """
    tracer = context.get_tracer_or_none()
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(
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
    **kwargs: typing.Any,
):
    """
    A decorator to trace function execution by recording a span for the traced function.

    Example:

    ```python
    import patronus

    patronus.init()

    # Trace a function with the @traced decorator
    @patronus.traced()
    def process_input(user_query):
        # Process the input
    ```

    Args:
        span_name (Optional[str]): The name of the traced span. Defaults to the function name if not provided.
        log_args (bool): Whether to log the arguments passed to the function. Default is True.
        log_results (bool): Whether to log the function's return value. Default is True.
        log_exceptions (bool): Whether to log any exceptions raised while executing the function. Default is True.
        disable_log (bool): Whether to disable logging the trace information. Default is False.
        attributes (Attributes): Attributes to attach to the traced span. Default is None.
        **kwargs: Additional arguments for the decorator.
    """

    def decorator(func):
        name = span_name or func.__qualname__
        sig = inspect.signature(func)
        record_exception = not disable_log and log_exceptions

        def log_call(fn_args: typing.Any, fn_kwargs: typing.Any, ret: typing.Any, exc: Exception):
            if disable_log:
                return

            logger = context.get_pat_logger()
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
            tracer = context.get_tracer_or_none()
            if tracer is None:
                return func(*f_args, **f_kwargs)

            exc = None
            ret = None
            with tracer.start_as_current_span(name, record_exception=record_exception, attributes=attributes):
                try:
                    ret = func(*f_args, **f_kwargs)
                except Exception as e:
                    exc = e
                    raise exc
                finally:
                    log_call(f_args, f_kwargs, ret, exc)

                return ret

        @functools.wraps(func)
        async def wrapper_async(*f_args, **f_kwargs):
            tracer = context.get_tracer_or_none()
            if tracer is None:
                return await func(*f_args, **f_kwargs)

            exc = None
            ret = None
            with tracer.start_as_current_span(name, record_exception=record_exception, attributes=attributes):
                try:
                    ret = await func(*f_args, **f_kwargs)
                except Exception as e:
                    exc = e
                    raise exc
                finally:
                    log_call(f_args, f_kwargs, ret, exc)

                return ret

        if inspect.iscoroutinefunction(func):
            wrapper_async._pat_traced = True
            return wrapper_async
        else:
            wrapper_async._pat_traced = True
            return wrapper_sync

    return decorator
