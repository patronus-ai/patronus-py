import functools
from typing import Optional

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from patronus import context
from patronus.tracing.attributes import Attributes, format_service_name


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
    exporter_endpoint: str,
    api_key: str,
    scope: context.PatronusScope,
) -> TracerProvider:
    service_name = format_service_name(scope.project_name, scope.app, scope.experiment_id)
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        PatronusAttributesSpanProcessor(
            project_name=scope.project_name,
            app=scope.app,
            experiment_id=scope.experiment_id,
        )
    )
    provider.add_span_processor(BatchSpanProcessor(_create_exporter(endpoint=exporter_endpoint, api_key=api_key)))
    return provider


def create_tracer(
    scope: context.PatronusScope,
    exporter_endpoint: str,
    api_key: str,
) -> trace.Tracer:
    provider = create_tracer_provider(
        exporter_endpoint=exporter_endpoint,
        api_key=api_key,
        scope=scope,
    )
    return provider.get_tracer("patronus.sdk")
