import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)

DEFAULT_BASE_OTEL_ENDPOINT = 'https://otel.patronus.ai:4317'


def init_tracer() -> trace.Tracer:
    trace_provider = TracerProvider()
    trace_processor = BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint=os.environ.get("BASE_OTEL_ENDPOINT", DEFAULT_BASE_OTEL_ENDPOINT),
            headers={"x-api-key": os.environ.get("PATRONUS_API_KEY")},
        )
    )
    trace_provider.add_span_processor(trace_processor)
    trace.set_tracer_provider(trace_provider)
    otel_tracer = trace_provider.get_tracer("PatronusTracer")
    return otel_tracer
