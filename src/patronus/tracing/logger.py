import os
from time import time_ns
from typing import Any, Optional

from opentelemetry._logs import SeverityNumber, set_logger_provider, get_logger
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as OTLPLogExporterTCP
from opentelemetry.sdk._logs import LoggerProvider as OTELLoggerProvider
from opentelemetry.sdk._logs._internal import LogRecord
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.trace import get_current_span
from opentelemetry.util.types import Attributes

BASE_OTEL_ENDPOINT = 'https://otel.patronus.ai:4317'
PAT_LOG_TYPE = "patronus-sdk-tracing"


class Logger:
    def __init__(
            self,
            project_name: str,
            app: str | None = None,
            experiment_id: str | None = None,
    ):
        self.project_name = project_name
        self.app = app
        self.experiment_id = experiment_id
        self.otel_logger = get_logger(project_name)

    def info(self, body: Any, attributes: Optional[Attributes] = None):
        self._log(
            serverity_number=SeverityNumber.INFO,
            body=body,
            attributes=attributes
        )

    def error(self, body: Any, attributes: Optional[Attributes] = None):
        self._log(
            serverity_number=SeverityNumber.ERROR,
            body=body,
            attributes=attributes
        )

    def debug(self, body: Any, attributes: Optional[Attributes] = None):
        self._log(
            serverity_number=SeverityNumber.DEBUG,
            body=body,
            attributes=attributes
        )

    def evaluation_log(self, *args, **kwargs):
        # TODO: @MJ
        raise NotImplementedError()

    def _log(self, serverity_number: SeverityNumber, body: Any, attributes: Optional[Attributes] = None):
        span_context = get_current_span().get_span_context()
        attributes = attributes or {}
        attributes["pat.log_type"] = PAT_LOG_TYPE

        if self.experiment_id:
            attributes["pat.experiment.id"] = self.experiment_id
        if self.app:
            attributes["pat.app"] = self.app
        self.otel_logger.emit(
            LogRecord(
                timestamp=time_ns(),
                observed_timestamp=time_ns(),
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                trace_flags=span_context.trace_flags,
                body=body,
                attributes=attributes,
                severity_number=serverity_number
            )
        )


def init_logger(project_name: str):
    processor = BatchLogRecordProcessor(
        OTLPLogExporterTCP(
            endpoint=os.getenv("BASE_OTEL_ENDPOINT", BASE_OTEL_ENDPOINT),
            headers={
                "x-api-key": os.getenv("PATRONUS_API_KEY"),
                "pat-project-name": project_name,
            },
            insecure=True
        )
    )
    logger_provider = OTELLoggerProvider()
    set_logger_provider(logger_provider)
    logger_provider.add_log_record_processor(processor)
    return Logger(project_name=project_name)
