import functools
from typing import Optional

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ..config import config
from ..context_utils import ContextObject
from .attributes import Attributes, format_service_name

DEFAULT_BASE_OTEL_ENDPOINT = "https://otel.patronus.ai:4317"


class PatronusAttributesSpanProcessor(SpanProcessor):
    project_name: str
    app: Optional[str]
    experiment_id: Optional[str]

    def __init__(self, project_name: str, app: Optional[str] = None, experiment_id: Optional[str] = None):
        self.project_name = project_name
        self.experiment_id = None
        self.app = None

        if experiment_id is not None:
            self.experiment_id = experiment_id
        else:
            self.app = app

    def on_start(self, span: Span, parent_context: Optional[context_api.Context] = None) -> None:
        attributes = {Attributes.project_name: self.project_name}
        if self.app is not None:
            attributes[Attributes.app] = self.app
        if self.experiment_id is not None:
            attributes[Attributes.experiment_id] = self.experiment_id

        span.set_attributes(attributes)
        super().on_start(span, parent_context)


@functools.lru_cache()
def _create_patronus_attributes_span_processor(
    project_name: str, app: Optional[str] = None, experiment_id: Optional[str] = None
):
    return PatronusAttributesSpanProcessor(project_name=project_name, app=app, experiment_id=experiment_id)


@functools.lru_cache()
def _create_exporter(endpoint: str, api_key: str) -> OTLPSpanExporter:
    return OTLPSpanExporter(endpoint=endpoint, headers={"x-api-key": api_key}, insecure=True)


@functools.lru_cache()
def create_tracer_provider(
    exporter_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
    app: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> TracerProvider:
    exporter_endpoint = exporter_endpoint or config().otel_endpoint
    api_key = api_key or config().api_key
    project_name = project_name or config().project_name
    if not experiment_id:
        app = app or config().app

    service_name = format_service_name(project_name, app, experiment_id)
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        PatronusAttributesSpanProcessor(project_name=project_name, app=app, experiment_id=experiment_id)
    )
    provider.add_span_processor(BatchSpanProcessor(_create_exporter(endpoint=exporter_endpoint, api_key=api_key)))
    return provider


def create_tracer(
    project_name: Optional[str] = None,
    app: Optional[str] = None,
    experiment_id: Optional[str] = None,
    exporter_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> trace.Tracer:
    provider = create_tracer_provider(
        exporter_endpoint=exporter_endpoint,
        api_key=api_key,
        project_name=project_name,
        app=app,
        experiment_id=experiment_id,
    )
    return provider.get_tracer("patronus.sdk")


_CTX_TRACER = ContextObject[trace.Tracer]("pat.tracer")


def get_tracer() -> trace.Tracer:
    return _CTX_TRACER.get()


#
# def init_tracer() -> trace.Tracer:
#     trace_provider = TracerProvider()
#     trace_processor = BatchSpanProcessor(
#         OTLPSpanExporter(
#             endpoint=os.environ.get("BASE_OTEL_ENDPOINT", DEFAULT_BASE_OTEL_ENDPOINT),
#             headers={"x-api-key": os.environ.get("PATRONUS_API_KEY")},
#         )
#     )
#     trace_provider.add_span_processor(trace_processor)
#     trace.set_tracer_provider(trace_provider)
#     otel_tracer = trace_provider.get_tracer("PatronusTracer")
#     return otel_tracer
