import inspect
from patronus.tracing import get_logger, get_tracer


def traced(*args, **kwargs):
    io_logging = not kwargs.pop("io_logging", False)
    project_name = kwargs.pop("project_name", "default")
    logger = kwargs.pop("logger", get_logger(project_name))
    tracer = kwargs.pop("tracer", get_tracer())
    def decorator(func):
        name = kwargs.pop("name") or args[0] if len(args) > 0 else func.__name__

        def wrapper_sync(*f_args, **f_kwargs):
            with tracer.start_as_current_span(name) as span:
                ret = func(*f_args, **f_kwargs)
                if io_logging:
                    argspec = inspect.getfullargspec(func)
                    if argspec.defaults:
                        positional_count = len(argspec.args) - len(argspec.defaults)
                        input_kwargs = dict(zip(argspec.args[positional_count:], argspec.defaults))
                        input_kwargs.update(f_kwargs)
                    else:
                        input_kwargs = f_kwargs
                    logger.info(
                        {
                            "input.args": f_args,
                            "input.kwargs": input_kwargs,
                            "output": str(ret)  # TODO: @MJ - serialize it?
                        }
                    )
                return ret

        async def wrapper_async(*f_args, **f_kwargs):
            with tracer.start_as_current_span(name) as span:
                ret = await func(*f_args, **f_kwargs)
                if io_logging:
                    argspec = inspect.getfullargspec(func)
                    if argspec.defaults:
                        positional_count = len(argspec.args) - len(argspec.defaults)
                        input_kwargs = dict(zip(argspec.args[positional_count:], argspec.defaults))
                        input_kwargs.update(f_kwargs)
                    else:
                        input_kwargs = f_kwargs
                    logger.info(
                        {
                            "input.args": f_args,
                            "input.kwargs": input_kwargs,
                            "output": str(ret)  # TODO: @MJ - serialize it?g
                        }
                    )
                return ret

        if inspect.iscoroutinefunction(func):
            return wrapper_async
        else:
            return wrapper_sync

    return decorator
