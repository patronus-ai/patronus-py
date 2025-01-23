import functools
import logging
import typing
import uuid
from enum import Enum
from time import time_ns
from types import MappingProxyType
from typing import Optional, Union

from opentelemetry._logs import SeverityNumber
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as OTLPLogExporterTCP
from opentelemetry.sdk._logs import Logger as OTELLogger
from opentelemetry.sdk._logs import LoggerProvider as OTELLoggerProvider
from opentelemetry.sdk._logs import LoggingHandler, LogRecord
from opentelemetry.sdk._logs._internal import ConcurrentMultiLogRecordProcessor, SynchronousMultiLogRecordProcessor
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import get_current_span
from opentelemetry.util.types import Attributes as OTeLAttributes

from patronus.config import config
from patronus.context_utils import ContextObject, ResourceMutex
from patronus.tracing.attributes import Attributes, format_service_name


class LogTypes(str, Enum):
    evaluation_data = "evaluation_data"
    unstructured_evaluation_data = "unstructured_evaluation_data"


class LoggerProvider(OTELLoggerProvider):
    project_name: Optional[str]
    app: Optional[str]
    experiment_id: Optional[str]

    def __init__(
        self,
        project_name: Optional[str] = None,
        app: Optional[str] = None,
        experiment_id: Optional[str] = None,
        shutdown_on_exit: bool = True,
        multi_log_record_processor: Union[
            SynchronousMultiLogRecordProcessor,
            ConcurrentMultiLogRecordProcessor,
        ] = None,
    ):
        self.project_name = project_name
        self.app = app
        self.experiment_id = experiment_id

        service_name = format_service_name(self.project_name, self.app, self.experiment_id)
        resource = Resource.create({"service.name": service_name})
        super().__init__(resource, shutdown_on_exit, multi_log_record_processor)

    def _get_logger_no_cache(
        self,
        name: str,
        version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Optional[OTeLAttributes] = None,
    ) -> "Logger":
        attributes = attributes or {}

        if Attributes.project_name not in attributes:
            attributes[Attributes.project_name] = self.project_name
        if Attributes.app not in attributes:
            attributes[Attributes.app] = self.app
        if Attributes.experiment_id not in attributes and self.experiment_id:
            attributes[Attributes.experiment_id] = self.experiment_id

        return Logger(
            self._resource,
            self._multi_log_record_processor,
            InstrumentationScope(
                name,
                version,
                schema_url,
                attributes,
            ),
        )

    def get_logger(self, *args, **kwargs) -> "Logger":
        return super().get_logger(*args, **kwargs)


def encode_attrs(v):
    if isinstance(v, dict):
        keys = list(v.keys())
        for k in keys:
            v[k] = encode_attrs(v[k])
        return v
    if isinstance(v, (list, tuple, str, int, float, bool, type(None))):
        return v
    return str(v)


def cleanup_log(v: typing.Any):
    # Logger cannot handle null values. Shouldn't be necessary according to the spec, but
    # the otel lib is not serializing nulls to proto (it seems like a omission/bug to me).
    if v is None:
        return "<null>"
    if isinstance(v, MappingProxyType):
        v = dict(v)
    if isinstance(v, list):
        return [cleanup_log(vv) for vv in v]
    if isinstance(v, tuple):
        return tuple(cleanup_log(vv) for vv in v)
    if isinstance(v, (str, bool, int, float)):
        return v
    if not isinstance(v, dict):
        return str(v)

    keys = list(v.keys())
    ret_v = {**v}
    for k in keys:
        ret_v[k] = cleanup_log(v[k])
    return ret_v


class Logger(OTELLogger):
    def __init__(
        self,
        resource: Resource,
        multi_log_record_processor: Union[
            SynchronousMultiLogRecordProcessor,
            ConcurrentMultiLogRecordProcessor,
        ],
        instrumentation_scope: InstrumentationScope,
    ):
        super().__init__(resource, multi_log_record_processor, instrumentation_scope)

    def log(
        self, body: typing.Any, log_attrs: Optional[OTeLAttributes] = None, severity: Optional[SeverityNumber] = None
    ) -> uuid.UUID:
        severity: SeverityNumber = severity or SeverityNumber.INFO
        span_context = get_current_span().get_span_context()
        log_id = uuid.uuid4()
        log_attrs = log_attrs or {}
        log_attrs[Attributes.log_id] = str(log_id)
        body = cleanup_log(body)
        self.emit(
            LogRecord(
                timestamp=time_ns(),
                observed_timestamp=time_ns(),
                # Invalid span IDs are 0, so set None instead in such case.
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                trace_flags=span_context.trace_flags,
                severity_text=severity.name,
                severity_number=severity,
                body=body,
                attributes=log_attrs,
                resource=self.resource,
            )
        )
        return log_id

    def evaluation_data(
        self,
        *,
        system_prompt: Optional[str] = None,
        task_context: Union[list[str], str, None] = None,
        task_input: Optional[str] = None,
        task_output: Optional[str] = None,
        gold_answer: Optional[str] = None,
        task_metadata: Optional[dict[str, typing.Any]] = None,
        log_attrs: Optional[OTeLAttributes] = None,
    ) -> uuid.UUID:
        if isinstance(task_context, str):
            task_context = [task_context]
        log_attrs = log_attrs or {}
        log_attrs[Attributes.log_type] = LogTypes.evaluation_data

        return self.log(
            {
                "system_prompt": system_prompt,
                "task_context": task_context,
                "task_input": task_input,
                "task_output": task_output,
                "gold_answer": gold_answer,
                "task_metadata": task_metadata,
            }
        )


@functools.lru_cache()
def _create_exporter(endpoint: str, api_key: str) -> OTLPLogExporterTCP:
    return OTLPLogExporterTCP(endpoint=endpoint, headers={"x-api-key": api_key}, insecure=True)


@functools.lru_cache()
def create_logger_provider(
    exporter_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
    app: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> LoggerProvider:
    exporter_endpoint = exporter_endpoint or config().otel_endpoint
    api_key = api_key or config().api_key
    project_name = project_name or config().project_name

    if not experiment_id:
        app = app or config().app

    logger_provider = LoggerProvider(project_name=project_name, app=app, experiment_id=experiment_id)
    exporter = _create_exporter(exporter_endpoint, api_key)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    return logger_provider


def create_patronus_logger(
    project_name: Optional[str] = None,
    app: Optional[str] = None,
    experiment_id: Optional[str] = None,
    exporter_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Logger:
    provider = create_logger_provider(
        exporter_endpoint=exporter_endpoint,
        api_key=api_key,
        project_name=project_name,
        app=app,
        experiment_id=experiment_id,
    )
    return provider.get_logger("patronus.sdk")


__logger_count = ResourceMutex(0)


@functools.lru_cache()
def create_logger(
    project_name: Optional[str] = None,
    app: Optional[str] = None,
    experiment_id: Optional[str] = None,
    exporter_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> logging.Logger:
    provider = create_logger_provider(
        exporter_endpoint=exporter_endpoint,
        api_key=api_key,
        project_name=project_name,
        app=app,
        experiment_id=experiment_id,
    )
    with __logger_count as mu:
        n = mu.get()
        mu.set(n + 1)
    if n == 0:
        suffix = ""
    else:
        suffix = f".{n}"
    logger = logging.getLogger(f"patronus.sdk{suffix}")
    logger.addHandler(LoggingHandler(level=logging.NOTSET, logger_provider=provider))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    return logger


_CTX_STD_LOGGER = ContextObject[logging.Logger]("pat.std-logger")
_CTX_EVAL_LOGGER = ContextObject[Logger]("pat.eval-logger")


def get_logger() -> logging.Logger:
    return _CTX_STD_LOGGER.get()


def get_patronus_logger() -> Logger:
    return _CTX_EVAL_LOGGER.get()
