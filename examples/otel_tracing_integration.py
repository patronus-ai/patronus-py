import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)

from patronus import Client

# Initialize Patronus Opentelemetry
trace_provider = TracerProvider()
trace_processor = BatchSpanProcessor(
    OTLPSpanExporter(
        endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        headers=os.environ.get("OTEL_EXPORTER_OTLP_HEADERS"),
    )
)
trace_provider.add_span_processor(trace_processor)
trace.set_tracer_provider(trace_provider)

# Get new tracer
tracer = trace.get_tracer("Demo Tracing")

# Initialize Patronus Client
client = Client(api_key=os.environ.get("PATRONUS_API_KEY"))

# Start Opentelemetry tracing
with tracer.start_as_current_span("span-name") as span:
    # Run evaluation
    result = client.evaluate(
        evaluator="lynx-small",
        criteria="patronus:hallucination",
        evaluated_model_input="What is the largest animal in the world?",
        evaluated_model_output="The giant sandworm.",
        evaluated_model_retrieved_context="The blue whale is the largest known animal.",
        tags={"scenario": "onboarding"},
    )
