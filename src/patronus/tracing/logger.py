import dataclasses
import functools
import logging
import typing
import uuid
from time import time_ns
from types import MappingProxyType
from typing import Optional, Union

from opentelemetry._logs import SeverityNumber
from patronus.tracing.exporters import create_log_exporter
from opentelemetry.sdk._logs import LogRecord
from opentelemetry.sdk._logs import Logger as OTELLogger
from opentelemetry.sdk._logs import LoggerProvider as OTELLoggerProvider
from opentelemetry.sdk._logs import LoggingHandler as OTeLLoggingHandler
from opentelemetry.sdk._logs._internal import ConcurrentMultiLogRecordProcessor, SynchronousMultiLogRecordProcessor
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import get_current_span, SpanContext
from opentelemetry.util.types import Attributes as OTeLAttributes

from patronus import context
from patronus.context.context_utils import ResourceMutex
from patronus.tracing.attributes import Attributes, LogTypes


@dataclasses.dataclass
class PatronusScope:
    project_name: Optional[str]
    app: Optional[str]
    experiment_id: Optional[str]
    experiment_name: Optional[str]

    @functools.cached_property
    def as_attributes(self) -> dict[str, str]:
        attrs = {}
        if self.project_name:
            attrs[Attributes.project_name.value] = self.project_name
        if self.app:
            attrs[Attributes.app.value] = self.app
        if self.experiment_id:
            attrs[Attributes.experiment_id.value] = self.experiment_id
        if self.experiment_name:
            attrs[Attributes.experiment_name.value] = self.experiment_name
        return attrs


class LoggerProvider(OTELLoggerProvider):
    project_name: Optional[str]
    app: Optional[str]
    experiment_id: Optional[str]
    experiment_name: Optional[str]

    def __init__(
        self,
        service: Optional[str] = None,
        project_name: Optional[str] = None,
        app: Optional[str] = None,
        experiment_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        shutdown_on_exit: bool = True,
        multi_log_record_processor: Union[
            SynchronousMultiLogRecordProcessor,
            ConcurrentMultiLogRecordProcessor,
        ] = None,
    ):
        self.project_name = project_name
        self.app = app
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name

        resource = None
        if service is not None:
            resource = Resource.create({"service.name": service})
        super().__init__(resource, shutdown_on_exit, multi_log_record_processor)

    def _get_logger_no_cache(
        self,
        name: str,
        version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Optional[OTeLAttributes] = None,
    ) -> "Logger":
        pat_scope = PatronusScope(
            project_name=self.project_name,
            app=self.app,
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
        )
        attributes = attributes or {}
        attributes = {**attributes, **pat_scope.as_attributes}
        return Logger(
            self._resource,
            self._multi_log_record_processor,
            InstrumentationScope(
                name,
                version,
                schema_url,
                attributes,
            ),
            pat_scope,
        )

    def get_logger(self, *args, **kwargs) -> "Logger":
        return super().get_logger(*args, **kwargs)


class LoggingHandler(OTeLLoggingHandler):
    def __init__(self, pat_scope: PatronusScope, level=logging.NOTSET, logger_provider=None) -> None:
        super().__init__(level, logger_provider)
        self.pat_scope = pat_scope

    def emit(self, record: logging.LogRecord) -> None:
        # Add patronus attributes to all outgoing logs.
        # This handler transforms record into OTeL LogRecord.
        for k, v in self.pat_scope.as_attributes.items():
            # We use setattr since attributes may have dots in their names
            # and record is an object, not a dict.
            setattr(record, k, v)
        setattr(record, Attributes.log_type, LogTypes.user.value)
        super().emit(record)


def encode_attrs(v):
    if isinstance(v, dict):
        keys = list(v.keys())
        for k in keys:
            v[k] = encode_attrs(v[k])
        return v
    if isinstance(v, (list, tuple, str, int, float, bool, type(None))):
        return v
    return str(v)


def transform_body(v: typing.Any):
    if v is None:
        return None
    if isinstance(v, MappingProxyType):
        v = dict(v)
    if isinstance(v, list):
        return [transform_body(vv) for vv in v]
    if isinstance(v, tuple):
        return tuple(transform_body(vv) for vv in v)
    if isinstance(v, (str, bool, int, float)):
        return v
    if not isinstance(v, dict):
        return str(v)

    return {str(k): transform_body(v) for k, v in v.items()}


class Logger(OTELLogger):
    def __init__(
        self,
        resource: Resource,
        multi_log_record_processor: Union[
            SynchronousMultiLogRecordProcessor,
            ConcurrentMultiLogRecordProcessor,
        ],
        instrumentation_scope: InstrumentationScope,
        pat_scope: PatronusScope,
    ):
        super().__init__(resource, multi_log_record_processor, instrumentation_scope)
        self.pat_scope = pat_scope

    def log(
        self,
        body: typing.Any,
        *,
        log_attrs: Optional[OTeLAttributes] = None,
        severity: Optional[SeverityNumber] = None,
        log_type: LogTypes = LogTypes.user,
        log_id: Optional[uuid.UUID] = None,
        span_context: Optional[SpanContext] = None,
        event_name: Optional[str] = None,
    ) -> uuid.UUID:
        severity: SeverityNumber = severity or SeverityNumber.INFO
        span_context = span_context or get_current_span().get_span_context()

        log_id = log_id or uuid.uuid4()
        log_attrs = log_attrs or {}
        log_attrs.update(
            {
                Attributes.log_id.value: str(log_id),
                Attributes.log_type.value: log_type.value,
            }
        )
        if event_name is not None:
            log_attrs[Attributes.event_name.value] = event_name
        log_attrs.update(self.pat_scope.as_attributes)

        record = LogRecord(
            timestamp=time_ns(),
            observed_timestamp=time_ns(),
            trace_id=span_context.trace_id,
            span_id=span_context.span_id,
            trace_flags=span_context.trace_flags,
            severity_text=severity.name,
            severity_number=severity,
            body=transform_body(body),
            attributes=log_attrs,
            resource=self.resource,
        )
        self.emit(record)
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
        log_attrs[Attributes.log_type] = LogTypes.eval

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


def _create_exporter(endpoint: str, api_key: str, protocol: Optional[str] = None):
    return create_log_exporter(endpoint=endpoint, api_key=api_key, protocol=protocol)


@functools.lru_cache()
def create_logger_provider(
    exporter_endpoint: str, api_key: str, scope: context.PatronusScope, protocol: Optional[str] = None
) -> LoggerProvider:
    logger_provider = LoggerProvider(project_name=scope.project_name, app=scope.app, experiment_id=scope.experiment_id)
    exporter = _create_exporter(exporter_endpoint, api_key, protocol)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    return logger_provider


def create_patronus_logger(
    scope: context.PatronusScope,
    exporter_endpoint: str,
    api_key: str,
    protocol: Optional[str] = None,
) -> Logger:
    provider = create_logger_provider(
        exporter_endpoint=exporter_endpoint,
        api_key=api_key,
        scope=scope,
        protocol=protocol,
    )
    return provider.get_logger("patronus.sdk")


__logger_count = ResourceMutex(0)


def set_logger_handler(logger: logging.Logger, scope: context.PatronusScope, provider: LoggerProvider):
    if any(isinstance(hdl, LoggingHandler) for hdl in logger.handlers):
        return
    scp = PatronusScope(
        project_name=scope.project_name,
        app=scope.app,
        experiment_id=scope.experiment_id,
        experiment_name=scope.experiment_name,
    )
    logger.addHandler(LoggingHandler(pat_scope=scp, level=logging.NOTSET, logger_provider=provider))


@functools.lru_cache()
def create_logger(
    scope: context.PatronusScope,
    exporter_endpoint: str,
    api_key: str,
    protocol: Optional[str] = None,
) -> logging.Logger:
    provider = create_logger_provider(
        exporter_endpoint=exporter_endpoint,
        api_key=api_key,
        scope=scope,
        protocol=protocol,
    )
    with __logger_count as mu:
        n = mu.get()
        mu.set(n + 1)
    if n == 0:
        suffix = ""
    else:
        suffix = f".{n}"
    logger = logging.getLogger(f"patronus.sdk{suffix}")
    pat_scope = PatronusScope(
        project_name=scope.project_name,
        app=scope.app,
        experiment_id=scope.experiment_id,
        experiment_name=scope.experiment_name,
    )
    logger.addHandler(LoggingHandler(pat_scope=pat_scope, level=logging.NOTSET, logger_provider=provider))
    logger.setLevel(logging.DEBUG)
    return logger
