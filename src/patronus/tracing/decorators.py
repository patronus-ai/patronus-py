import functools
import inspect

from .logger import get_patronus_logger
from .trace import get_tracer


def traced(*args, **kwargs):
    io_logging = not kwargs.pop("io_logging", False)

    def decorator(func):
        name = kwargs.pop("name") or args[0] if len(args) > 0 else func.__name__
        ignore_input = bool(kwargs.get("ignore_input", False))
        ignore_output = bool(kwargs.get("ignore_output", False))

        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper_sync(*f_args, **f_kwargs):
            logger = get_patronus_logger()
            tracer = get_tracer()

            with tracer.start_as_current_span(name):
                ret = func(*f_args, **f_kwargs)
                if io_logging:
                    log_data = {"name": name}
                    if not ignore_input:
                        bound_args = sig.bind(*f_args, **f_kwargs)
                        log_data.update({"input.arguments": {**bound_args.arguments, **bound_args.kwargs}})
                    if not ignore_output:
                        log_data["output"] = str(ret)
                    if log_data:
                        logger.log(log_data)
                return ret

        @functools.wraps(func)
        async def wrapper_async(*f_args, **f_kwargs):
            logger = get_patronus_logger()
            tracer = get_tracer()

            with tracer.start_as_current_span(name):
                ret = await func(*f_args, **f_kwargs)
                if io_logging:
                    log_data = {"name": name}
                    if not ignore_input:
                        bound_args = sig.bind(*f_args, **f_kwargs)
                        log_data.update({"input.arguments": {**bound_args.arguments, **bound_args.kwargs}})
                    if not ignore_output:
                        log_data["output"] = str(ret)
                    if log_data:
                        logger.log(log_data)
                return ret

        if inspect.iscoroutinefunction(func):
            return wrapper_async
        else:
            return wrapper_sync

    return decorator
