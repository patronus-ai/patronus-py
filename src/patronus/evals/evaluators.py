import abc
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import threading
import time
import typing
import uuid
from opentelemetry.trace import get_current_span, SpanContext
from typing import Any, Optional, Union

from patronus import context
from patronus.api.api_client import PatronusAPIClient
from patronus.evals.context import get_context_evaluation_attributes
from patronus.api import api_types
from patronus.evals.types import EvaluationResult
from patronus.exceptions import UninitializedError
from patronus.retry import retry
from patronus.tracing.attributes import LogTypes, GenAIAttributes, OperationNames
from patronus.tracing.decorators import start_span
from patronus.tracing.logger import Logger
from patronus.utils import merge_tags

log = logging.getLogger("patronus.core")


LogID = uuid.UUID


class EvaluationDataRecord(typing.NamedTuple):
    """
    EvaluationDataRecord holds evaluation log data and ID of the log.
    Log data consists of positional arguments (args) and keyword arguments (kwargs)
    that were passed to an evaluator.
    """

    log_id: LogID
    arguments: dict

    def __str__(self):
        return str({"log_id": self.log_id, "arguments": self.arguments})


class UniqueEvaluationDataSet:
    """
    UniqueEvaluationDataSet holds unique list evaluation log data.
    This container is used to track and ensure that duplicated evaluation data logs are not emitted.
    """

    span_context: SpanContext
    logs: typing.List[EvaluationDataRecord]
    lock: threading.Lock

    __slots__ = ("span_context", "logs", "lock")

    def __init__(self, span_context: SpanContext):
        self.span_context = span_context
        self.logs = []
        self.lock = threading.Lock()

    def __str__(self):
        return str({"id": id(self), "span_context": self.span_context, "logs": self.logs})

    def __enter__(self) -> "UniqueEvaluationDataSet":
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

    def find_log(self, bound_arguments) -> Optional[LogID]:
        """Must be used with lock"""
        for lg in self.logs:
            if bound_arguments == lg.arguments:
                return lg.log_id
        return None

    def _add_log(self, arguments: dict[str, Any]) -> EvaluationDataRecord:
        """Must be used with lock"""
        eval_log = EvaluationDataRecord(
            log_id=uuid.uuid4(),
            arguments=arguments,
        )
        self.logs = [*self.logs, eval_log]
        return eval_log

    def log(
        self, *, logger: Logger, is_method: bool, self_key_name: Optional[str], bound_arguments: dict[str, Any]
    ) -> LogID:
        if is_method:
            assert self_key_name is not None
            bound_arguments = {**bound_arguments}
            bound_arguments.pop(self_key_name)
        with self.lock:
            log_id = self.find_log(bound_arguments)
            if log_id is not None:
                return log_id
            record = self._add_log(bound_arguments)
        logger.log(
            body=record.arguments,
            log_type=LogTypes.eval,
            log_id=record.log_id,
            span_context=self.span_context,
        )
        return record.log_id


_ctx_evaluation_log_group: contextvars.ContextVar[UniqueEvaluationDataSet] = contextvars.ContextVar("_ctx_bundled_eval")


def get_current_log_id(bound_arguments: dict[str, Any]) -> Optional[LogID]:
    """
    Return log_id for given arguments in current context.
    Returns None if there is no context - most likely SDK is not initialized.
    """
    eval_group = _ctx_evaluation_log_group.get(None)
    if eval_group is None:
        return None
    log_id = eval_group.find_log(bound_arguments)
    if log_id is None:
        raise ValueError("Log not found for provided arguments")
    return log_id


@contextlib.contextmanager
def _start_evaluation_log_group() -> typing.Iterator[UniqueEvaluationDataSet]:
    span = get_current_span()
    value = UniqueEvaluationDataSet(span_context=span.get_span_context())
    tokens = _ctx_evaluation_log_group.set(value)
    yield value
    _ctx_evaluation_log_group.reset(tokens)


@contextlib.contextmanager
def _get_or_start_evaluation_log_group() -> typing.Iterator[UniqueEvaluationDataSet]:
    ctx = _ctx_evaluation_log_group.get(None)
    if ctx is not None:
        yield ctx
    else:
        with _start_evaluation_log_group() as ctx:
            yield ctx


@contextlib.contextmanager
def bundled_eval(span_name: str = "Evaluation bundle"):
    tracer = context.get_tracer_or_none()
    if tracer is None:
        yield
        return

    with tracer.start_as_current_span(span_name):
        with _start_evaluation_log_group():
            yield


def handle_eval_output(
    *,
    ctx: context.PatronusContext,
    log_id: LogID,
    evaluator_id: str,
    criteria: Optional[str] = None,
    metric_name: str,
    metric_description: str,
    ret_value: typing.Any,
    duration: datetime.timedelta,
    qualname: str,
):
    # Returned None means no evaluation took place.
    if ret_value is None:
        return

    ev: EvaluationResult = coerce_eval_output_type(ret_value, qualname)

    span_context = get_current_span().get_span_context()
    evaluation_attrs = get_context_evaluation_attributes()
    tags = merge_tags(evaluation_attrs["tags"], ev.tags, evaluation_attrs["experiment_tags"])
    try:
        eval_payload = api_types.ClientEvaluation(
            log_id=log_id,
            project_name=ctx.scope.project_name,
            app=ctx.scope.app,
            experiment_id=ctx.scope.experiment_id,
            evaluator_id=evaluator_id,
            criteria=criteria,
            pass_=ev.pass_,
            score=ev.score,
            text_output=ev.text_output,
            metadata=ev.metadata,
            explanation=ev.explanation,
            explanation_duration=ev.explanation_duration,
            evaluation_duration=ev.evaluation_duration or duration,
            metric_name=metric_name,
            metric_description=metric_description,
            dataset_id=evaluation_attrs["dataset_id"],
            dataset_sample_id=evaluation_attrs["dataset_sample_id"],
            created_at=datetime.datetime.now(datetime.timezone.utc),
            tags=tags,
            trace_id=hex(span_context.trace_id)[2:].zfill(32),
            span_id=hex(span_context.span_id)[2:].zfill(16),
        )
        ctx.exporter.submit(eval_payload)
    except Exception:
        log.exception("Failed to submit evaluation payload to the exported")

    # # Debug evaluation log
    # ev_data = sorted([f"{k}={v!r}" for k, v in ev.model_dump().items() if v is not None])
    # get_logger().warning(f"Evaluation({', '.join(ev_data)})")


def coerce_eval_output_type(ev_output: typing.Any, qualname: str) -> EvaluationResult:
    if isinstance(ev_output, EvaluationResult):
        return ev_output
    if isinstance(ev_output, bool):
        return EvaluationResult(pass_=ev_output, score=float(ev_output))
    if isinstance(ev_output, (int, float)):
        return EvaluationResult(score=float(ev_output))
    if isinstance(ev_output, str):
        return EvaluationResult(text_output=ev_output)

    raise TypeError(
        f"Evaluator '{qualname}' returned unexpected type {type(ev_output)!r}. "
        f"Supported return types are EvaluationResult, int, float, bool, str. "
    )


class PrepEval(typing.NamedTuple):
    evaluator_id: str
    criteria: Optional[str]
    metric_name: str
    metric_description: Optional[str]
    self_key_name: Optional[str]
    arguments: dict[str, typing.Any]
    disable_export: bool

    def display_name(self):
        if not self.criteria:
            return self.evaluator_id
        return f"{self.criteria} ({self.evaluator_id})"


def _as_applied_argument(signature: inspect.Signature, bound_arguments: inspect.BoundArguments):
    """
    This function is very similar to inspect.BoundArguments.apply_default().

    The difference is that it's not meant to be used for arguments bounding but for as a log record of evaluation data.
    The purpose of it is to provide nicer display for arguments passed to evaluation functions.
    This function will ignore empty varied positional arguments (*args) and varied keyword arguments (**kwargs)
    if are empty (were not provided by the function called).
    """
    arguments = bound_arguments.arguments
    new_arguments = []
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            for kwname, kwvalue in arguments.get(name, {}).items():
                new_arguments.append((kwname, kwvalue))
        else:
            try:
                new_arguments.append((name, arguments[name]))
            except KeyError:
                if param.default is not inspect.Parameter.empty:
                    val = param.default
                else:
                    continue
                new_arguments.append((name, val))
    return dict(new_arguments)


def evaluator(
    *,
    # Name for the evaluator. Defaults to function name (or class name in case of class based evaluators).
    evaluator_id: Union[str, typing.Callable[[], str], None] = None,
    # Name of the criteria used by the evaluator.
    # The use of the criteria is only recommended in more complex evaluator setups
    # where evaluation algorithm changes depending on a criteria (think strategy pattern).
    criteria: Union[str, typing.Callable[[], str], None] = None,
    # Name for the evaluation metric. Defaults to evaluator_id value.
    metric_name: Optional[str] = None,
    # The description of the metric used for evaluation.
    # If not provided then the docstring of the wrapped function is used for this value.
    metric_description: Optional[str] = None,
    # Whether the wrapped function is a method.
    # This value is used to determine whether to remove "self" argument from the log.
    # It also allows for dynamic evaluator_id and criteria discovery
    # based on `get_evaluator_id()` and `get_criteria_id()` methods.
    # User-code usually shouldn't use it as long as user defined class-based evaluators inherit from
    # the library provided Evaluator base classes.
    is_method: bool = False,
    **kwargs,
):
    def decorator(fn):
        fn_sign = inspect.signature(fn)

        def _get_eval_id():
            return (callable(evaluator_id) and evaluator_id()) or evaluator_id or fn.__qualname__

        def _get_criteria():
            return (callable(criteria) and criteria()) or criteria or None

        def _prep(*fn_args, **fn_kwargs):
            bound_args = fn_sign.bind(*fn_args, **fn_kwargs)
            arguments_to_log = _as_applied_argument(fn_sign, bound_args)
            bound_args.apply_defaults()
            self_key_name = None
            instance = None
            if is_method:
                self_key_name = next(iter(fn_sign.parameters.keys()))
                instance = bound_args.arguments[self_key_name]

            eval_id = None
            eval_criteria = None
            if isinstance(instance, Evaluator):
                eval_id = instance.get_evaluator_id()
                eval_criteria = instance.get_criteria()

            if eval_id is None:
                eval_id = _get_eval_id()
            if eval_criteria is None:
                eval_criteria = _get_criteria()

            met_name = metric_name or eval_id
            met_description = metric_description or inspect.getdoc(fn) or None

            disable_export = isinstance(instance, RemoteEvaluatorMixin) and instance._disable_export

            return PrepEval(
                evaluator_id=eval_id,
                criteria=eval_criteria,
                metric_name=met_name,
                metric_description=met_description,
                self_key_name=self_key_name,
                arguments=arguments_to_log,
                disable_export=disable_export,
            )

        @functools.wraps(fn)
        async def wrapper_async(*fn_args, **fn_kwargs):
            ctx = context.get_current_context_or_none()
            if ctx is None:
                return await fn(*fn_args, **fn_kwargs)

            prep = _prep(*fn_args, **fn_kwargs)

            start = time.perf_counter()
            try:
                with start_span(prep.display_name(), attributes={GenAIAttributes.operation_name: OperationNames.eval}):
                    with _get_or_start_evaluation_log_group() as log_group:
                        log_id = log_group.log(
                            logger=ctx.pat_logger,
                            is_method=is_method,
                            self_key_name=prep.self_key_name,
                            bound_arguments=prep.arguments,
                        )
                        ret = await fn(*fn_args, **fn_kwargs)
            except Exception as e:
                ctx.logger.exception(f"Evaluator raised an exception: {e}")
                raise e
            if prep.disable_export:
                return ret
            elapsed = time.perf_counter() - start
            handle_eval_output(
                ctx=ctx,
                log_id=log_id,
                evaluator_id=prep.evaluator_id,
                criteria=prep.criteria,
                metric_name=prep.metric_name,
                metric_description=prep.metric_description,
                ret_value=ret,
                duration=datetime.timedelta(seconds=elapsed),
                qualname=fn.__qualname__,
            )
            return ret

        @functools.wraps(fn)
        def wrapper_sync(*fn_args, **fn_kwargs):
            ctx = context.get_current_context_or_none()
            if ctx is None:
                return fn(*fn_args, **fn_kwargs)

            prep = _prep(*fn_args, **fn_kwargs)

            start = time.perf_counter()
            try:
                with start_span(prep.display_name(), attributes={GenAIAttributes.operation_name: OperationNames.eval}):
                    with _get_or_start_evaluation_log_group() as log_group:
                        log_id = log_group.log(
                            logger=ctx.pat_logger,
                            is_method=is_method,
                            self_key_name=prep.self_key_name,
                            bound_arguments=prep.arguments,
                        )
                        ret = fn(*fn_args, **fn_kwargs)
            except Exception as e:
                ctx.logger.exception("Evaluation failed")
                raise e
            if prep.disable_export:
                return ret
            elapsed = time.perf_counter() - start
            handle_eval_output(
                ctx=ctx,
                log_id=log_id,
                evaluator_id=prep.evaluator_id,
                criteria=prep.criteria,
                metric_name=prep.metric_name,
                metric_description=prep.metric_description,
                ret_value=ret,
                duration=datetime.timedelta(seconds=elapsed),
                qualname=fn.__qualname__,
            )
            return ret

        def _set_attrs(wrapper: Any):
            wrapper._pat_evaluator = True

            # _pat_evaluator_id and _pat_criteria_id may be a bit misleading since
            # may not be correct since actually values for evaluator_id and criteria
            # are dynamically dispatched for class-based evaluators.
            # These values will be correct for function evaluators though.
            wrapper._pat_evaluator_id = _get_eval_id()
            wrapper._pat_criteria = _get_criteria()

        if inspect.iscoroutinefunction(fn):
            _set_attrs(wrapper_async)
            return wrapper_async
        else:
            _set_attrs(wrapper_sync)
            return wrapper_sync

    return decorator


# Inherit from ABCMeta so we can enforce abstractmethods
class _EvaluatorMeta(abc.ABCMeta):
    def __new__(mcls, name, bases, namespace, /, **kwargs):
        if (
            "evaluate" in namespace
            and
            # Skip wrapping in case the class is still an abstract class
            getattr(namespace["evaluate"], "__isabstractmethod__", None) is not True
        ):
            # Wrap evaluate method with evaluator decorator as soon as class is created.
            # This happens on class creation, not class instantiation.
            evaluator_id = namespace.get("evaluator_id", name)
            into_evaluator = evaluator(
                evaluator_id=evaluator_id,
                criteria=namespace.get("criteria"),
                is_method=True,
            )
            namespace["evaluate"] = into_evaluator(namespace["evaluate"])
        return super().__new__(mcls, name, bases, namespace, **kwargs)


class Evaluator(metaclass=_EvaluatorMeta):
    evaluator_id: Optional[str] = None
    criteria: Optional[str] = None

    def get_evaluator_id(self) -> str:
        return self.evaluator_id or self.__class__.__qualname__

    def get_criteria(self) -> Optional[str]:
        return self.criteria

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs) -> Optional[EvaluationResult]:
        """
        Synchronous version of evaluate method.
        When inheriting directly from Evaluator class it's permitted to change parameters signature.
        Return type should stay unchanged.
        """


class AsyncEvaluator(Evaluator):
    @abc.abstractmethod
    async def evaluate(self, *args, **kwargs) -> Optional[EvaluationResult]:
        """
        Asynchronous version of evaluate method.
        When inheriting directly from Evaluator class it's permitted to change parameters signature.
        Return type should stay unchanged.
        """


class StructuredEvaluator(Evaluator):
    @abc.abstractmethod
    def evaluate(
        self,
        *,
        system_prompt: Optional[str] = None,
        task_context: Union[list[str], str, None] = None,
        task_attachments: Union[list[Any], None] = None,
        task_input: Optional[str] = None,
        task_output: Optional[str] = None,
        gold_answer: Optional[str] = None,
        task_metadata: Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: Any,
    ) -> EvaluationResult: ...


class AsyncStructuredEvaluator(AsyncEvaluator):
    @abc.abstractmethod
    async def evaluate(
        self,
        *,
        system_prompt: Optional[str] = None,
        task_context: Union[list[str], str, None] = None,
        task_attachments: Union[list[Any], None] = None,
        task_input: Optional[str] = None,
        task_output: Optional[str] = None,
        gold_answer: Optional[str] = None,
        task_metadata: Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: Any,
    ) -> EvaluationResult: ...


class RemoteEvaluatorMixin:
    _disable_export = True

    def __init__(
        self,
        evaluator_id_or_alias: str,
        criteria: Optional[str] = None,
        *,
        tags: Optional[dict[str, str]] = None,
        explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"] = "always",
        criteria_config: Optional[dict[str, typing.Any]] = None,
        allow_update: bool = False,
        max_attempts: int = 3,
        api_: Optional[PatronusAPIClient] = None,
    ):
        self.evaluator_id_or_alias = evaluator_id_or_alias
        self.criteria = criteria
        self.tags = tags or {}
        self.explain_strategy = explain_strategy
        self.criteria_config = criteria_config
        self.allow_update = allow_update
        self.max_attempts = max_attempts
        self._api = api_

    def get_evaluator_id(self) -> str:
        return self.evaluator_id_or_alias

    def get_criteria(self) -> str:
        return self.criteria

    def _get_api(self) -> PatronusAPIClient:
        api_client = self._api or context.get_api_client_or_none()
        if api_client is None:
            raise UninitializedError(
                "Failed to get API client. Either initialize the library by calling patronus.init() or "
                "Apss the API client object to the RemoveEvaluator constructor."
            )
        return api_client

    @staticmethod
    def _translate_response(resp: api_types.EvaluationResult) -> EvaluationResult:
        return EvaluationResult(
            score=resp.score_raw,
            pass_=resp.pass_,
            text_output=resp.text_output,
            metadata=resp.additional_info,
            explanation=resp.explanation,
            tags=resp.tags,
            evaluation_duration=resp.evaluation_duration,
            explanation_duration=resp.explanation_duration,
        )


class RemoteEvaluator(RemoteEvaluatorMixin, StructuredEvaluator):
    def evaluate(
        self,
        *,
        system_prompt: Optional[str] = None,
        task_context: Union[list[str], str, None] = None,
        task_attachments: Union[list[Any], None] = None,
        task_input: Optional[str] = None,
        task_output: Optional[str] = None,
        gold_answer: Optional[str] = None,
        task_metadata: Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        kws = {
            "system_prompt": system_prompt,
            "task_context": task_context,
            "task_attachments": task_attachments,
            "task_input": task_input,
            "task_output": task_output,
            "gold_answer": gold_answer,
            "task_metadata": task_metadata,
            **kwargs,
        }
        log_id = get_current_log_id(bound_arguments=kws)

        attrs = get_context_evaluation_attributes()
        tags = {**self.tags}
        if t := attrs["tags"]:
            tags.update(t)
        tags = merge_tags(tags, kwargs.get("tags"), attrs["experiment_tags"])
        if tags:
            kws["tags"] = tags
        if did := attrs["dataset_id"]:
            kws["dataset_id"] = did
        if sid := attrs["dataset_sample_id"]:
            kws["dataset_sample_id"] = sid

        resp = retry()(self._evaluate)(log_id=log_id, **kws)
        return self._translate_response(resp)

    def _evaluate(
        self,
        *,
        log_id: Optional[LogID] = None,
        system_prompt: Optional[str] = None,
        task_context: Union[list[str], str, None] = None,
        task_attachments: Union[list[Any], None] = None,
        task_input: Optional[str] = None,
        task_output: Optional[str] = None,
        gold_answer: Optional[str] = None,
        task_metadata: Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: Any,
    ) -> api_types.EvaluationResult:
        project_name = None
        app = None
        experiment_id = None
        ctx = context.get_current_context_or_none()
        if ctx is not None:
            project_name = ctx.scope.project_name
            app = ctx.scope.app
            experiment_id = ctx.scope.experiment_id

        span_context = get_current_span().get_span_context()
        return self._get_api().evaluate_one_sync(
            api_types.EvaluateRequest(
                evaluators=[
                    api_types.EvaluateEvaluator(
                        evaluator=self.evaluator_id_or_alias,
                        criteria=self.criteria,
                        explain_strategy=self.explain_strategy,
                    )
                ],
                evaluated_model_system_prompt=system_prompt,
                evaluated_model_retrieved_context=task_context,
                evaluated_model_input=task_input,
                evaluated_model_output=task_output,
                evaluated_model_gold_answer=gold_answer,
                evaluated_model_attachments=task_attachments,
                project_id=None,
                project_name=project_name,
                app=app,
                experiment_id=experiment_id,
                capture="all",
                dataset_id=kwargs.get("dataset_id"),
                dataset_sample_id=kwargs.get("dataset_sample_id"),
                tags=kwargs.get("tags"),
                trace_id=hex(span_context.trace_id)[2:].zfill(32),
                span_id=hex(span_context.span_id)[2:].zfill(16),
                log_id=log_id and str(log_id),
            )
        )


class AsyncRemoteEvaluator(RemoteEvaluatorMixin, AsyncStructuredEvaluator):
    async def evaluate(
        self,
        *,
        system_prompt: Optional[str] = None,
        task_context: Union[list[str], str, None] = None,
        task_attachments: Union[list[Any], None] = None,
        task_input: Optional[str] = None,
        task_output: Optional[str] = None,
        gold_answer: Optional[str] = None,
        task_metadata: Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        kws = {
            "system_prompt": system_prompt,
            "task_context": task_context,
            "task_attachments": task_attachments,
            "task_input": task_input,
            "task_output": task_output,
            "gold_answer": gold_answer,
            "task_metadata": task_metadata,
            **kwargs,
        }
        log_id = get_current_log_id(bound_arguments=kws)

        attrs = get_context_evaluation_attributes()
        tags = {**self.tags}
        if t := attrs["tags"]:
            tags.update(t)
        tags = merge_tags(tags, kwargs.get("tags"), attrs["experiment_tags"])
        if tags:
            kws["tags"] = tags
        if did := attrs["dataset_id"]:
            kws["dataset_id"] = did
        if sid := attrs["dataset_sample_id"]:
            kws["dataset_sample_id"] = sid

        resp = await retry()(self._evaluate)(log_id=log_id, **kws)
        return self._translate_response(resp)

    async def _evaluate(
        self,
        *,
        log_id: Optional[LogID] = None,
        system_prompt: Optional[str] = None,
        task_context: Union[list[str], str, None] = None,
        task_attachments: Union[list[Any], None] = None,
        task_input: Optional[str] = None,
        task_output: Optional[str] = None,
        gold_answer: Optional[str] = None,
        task_metadata: Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: Any,
    ) -> api_types.EvaluationResult:
        project_name = None
        app = None
        experiment_id = None
        ctx = context.get_current_context_or_none()
        if ctx is not None:
            project_name = ctx.scope.project_name
            app = ctx.scope.app
            experiment_id = ctx.scope.experiment_id

        span_context = get_current_span().get_span_context()
        return await self._get_api().evaluate_one(
            api_types.EvaluateRequest(
                evaluators=[
                    api_types.EvaluateEvaluator(
                        evaluator=self.evaluator_id_or_alias,
                        criteria=self.criteria,
                        explain_strategy=self.explain_strategy,
                    )
                ],
                evaluated_model_system_prompt=system_prompt,
                evaluated_model_retrieved_context=task_context,
                evaluated_model_input=task_input,
                evaluated_model_output=task_output,
                evaluated_model_gold_answer=gold_answer,
                evaluated_model_attachments=task_attachments,
                project_id=None,
                project_name=project_name,
                app=app,
                experiment_id=experiment_id,
                capture="all",
                dataset_id=kwargs.get("dataset_id"),
                dataset_sample_id=kwargs.get("dataset_sample_id"),
                tags=kwargs.get("tags"),
                trace_id=hex(span_context.trace_id)[2:].zfill(32),
                span_id=hex(span_context.span_id)[2:].zfill(16),
                log_id=log_id and str(log_id),
            )
        )
