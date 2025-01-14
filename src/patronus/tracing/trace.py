import inspect
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)

from patronus.tracing.logger import get_patronus_logger

BASE_OTEL_ENDPOINT = 'https://otel.patronus.ai:4317'

trace_provider = TracerProvider()
trace_processor = BatchSpanProcessor(
    OTLPSpanExporter(
        endpoint=os.environ.get("BASE_OTEL_ENDPOINT", BASE_OTEL_ENDPOINT),
        headers={"x-api-key": os.environ.get("PATRONUS_API_KEY")},
    )
)
trace_provider.add_span_processor(trace_processor)
trace.set_tracer_provider(trace_provider)
otel_tracer = trace_provider.get_tracer("PatronusTracer")


def traced(*args, **kwargs):
    io_logging = not kwargs.pop("io_logging", False)
    project_name = kwargs.pop("project_name", "default")
    logger = get_patronus_logger("PatronusLogger", project_name)

    def decorator(func):
        name = kwargs.pop("name") or args[0] if len(args) > 0 else func.__name__

        def wrapper_sync(*f_args, **f_kwargs):
            with otel_tracer.start_as_current_span(name) as span:
                ret = func(*f_args, **f_kwargs)
                if io_logging:
                    logger.log(
                        {
                            "input.args": f_args,
                            "input.kwargs": f_kwargs,
                            "output": str(ret)
                        }
                    )
                return ret

        async def wrapper_async(*f_args, **f_kwargs):
            with otel_tracer.start_as_current_span(name) as span:
                print('Tracing', span)
                ret = await func(*f_args, **f_kwargs)
                if io_logging:
                    logger.log(
                        {
                            "input.args": f_args,
                            "input.kwargs": f_kwargs,
                            "output": str(ret)
                        }
                    )
                return ret

        if inspect.iscoroutinefunction(func):
            return wrapper_async
        else:
            return wrapper_sync

    return decorator
