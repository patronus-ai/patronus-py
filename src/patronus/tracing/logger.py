import os
from time import time_ns
from typing import Any, Optional

from opentelemetry._logs import set_logger_provider, SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider, LogRecord
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as OTLPLogExporterHTTP
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as OTLPLogExporterTCP
from opentelemetry.sdk._logs._internal.export import BatchLogRecordProcessor
from opentelemetry.trace import get_current_span
from opentelemetry.util.types import Attributes, AttributeValue

logger_provider = LoggerProvider()
set_logger_provider(logger_provider)


BASE_OTEL_ENDPOINT = 'https://otel.patronus.ai:4317'


exporter_rcp = OTLPLogExporterTCP(
    endpoint=os.getenv("BASE_OTEL_ENDPOINT", BASE_OTEL_ENDPOINT),
    headers={"x-api-key": os.getenv("PATRONUS_API_KEY")},
    insecure=True
)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter_rcp))


class Logger():
    _loggers = {}

    def __new__(cls, name: str, *args, **kwargs):
        if obj := cls._loggers.get(name):
            return obj
        obj = super().__new__(cls)
        cls._loggers[name] = obj
        return obj

    def __init__(self, name: str, project_id: str, app: str | None = None, experiment_id: str | None = None):
        self._name = name
        self.project_id = project_id
        self.app = app
        self.experiment_id = experiment_id
        self._loggers[name] = self

    def log(self, body: Any, attributes: Optional[Attributes] = None):
        logger = logger_provider.get_logger(
            self._name,
        )
        span_context = get_current_span().get_span_context()
        attributes = attributes or {}
        attributes["pat.project.id"] = self.project_id
        attributes["pat.log_type"] = "patronus-sdk-tracing"

        if self.experiment_id:
            attributes["pat.experiment.id"] = self.experiment_id
        if self.app:
            attributes["pat.app"] = self.app

        log_record = LogRecord(
            timestamp=time_ns(),
            observed_timestamp=time_ns(),
            trace_id=span_context.trace_id,
            span_id=span_context.span_id,
            trace_flags=span_context.trace_flags,
            body=body,
            attributes=attributes,
            severity_number=SeverityNumber.INFO
        )
        logger.emit(record=log_record)


def get_patronus_logger(name: str, project_id: str):
    return Logger(name=name, project_id=project_id)
