import abc
import datetime
import functools
import inspect
import logging
import re
import time
import typing
from typing import Any, Optional, Union

from opentelemetry.trace import get_current_span

from patronus import api_types
from patronus.evals.exporter import get_exporter
from patronus.evals.types import EvaluationResult
from patronus.tracing.attributes import Attributes
from patronus.tracing.decorators import traced
from patronus.tracing.logger import get_logger, get_patronus_logger

log = logging.getLogger("patronus.core")


def create_field_sanitizer(pattern: str, max_len: int, replace_with: str):
    def sanitize(value: str) -> str:
        return re.sub(pattern, replace_with, value[:max_len])

    return sanitize


sanitize_project_name = create_field_sanitizer(r"[^a-zA-Z0-9_ -]", max_len=50, replace_with="_")
sanitize_app = create_field_sanitizer(r"[^a-zA-Z0-9-_./ -]", max_len=50, replace_with="_")


def handle_eval_output(
    *,
    class_name: Optional[str],
    fn_name: str,
    evaluator_id: str,
    metric_name: str,
    metric_description: str,
    fn_sign: inspect.Signature,
    args,
    kwargs,
    ret_value: typing.Any,
    duration: datetime.timedelta,
):
    if ret_value is None:
        return

    ev: EvaluationResult = coerce_eval_output_type(ret_value, class_name, fn_name)
    pat_logger = get_patronus_logger()
    span_context = get_current_span().get_span_context()
    bound_args = fn_sign.bind(*args, **kwargs)
    log_id = pat_logger.log(
        body={"evaluation_data": {**bound_args.arguments, **bound_args.kwargs}},
    )

    # TODO there should be a better way of extracting project name and an app from execution context
    project_name = pat_logger._instrumentation_scope.attributes.get(Attributes.project_name)
    if project_name is not None:
        project_name = sanitize_project_name(project_name)
    app = pat_logger._instrumentation_scope.attributes.get(Attributes.app)
    if app is not None:
        app = sanitize_app(app)

    try:
        eval_payload = api_types.ClientEvaluation(
            log_id=log_id,
            project_name=project_name,
            app=app,
            experiment_id=pat_logger._instrumentation_scope.attributes.get(Attributes.experiment_id),
            evaluator_id=evaluator_id,
            criteria=None,  # TODO
            pass_=ev.pass_,
            score=ev.score,
            text_output=ev.text_output,
            metadata=ev.metadata,
            explanation=ev.explanation,
            explanation_duration=ev.explanation_duration,
            evaluation_duration=ev.evaluation_duration or duration,
            metric_name=metric_name,
            metric_description=metric_description,
            dataset_id=None,  # TODO
            dataset_sample_id=None,  # TODO
            created_at=datetime.datetime.now(datetime.timezone.utc),
            tags=ev.tags,
            trace_id=hex(span_context.trace_id)[2:].zfill(32),
            span_id=hex(span_context.span_id)[2:].zfill(16),
        )
        get_exporter().submit(eval_payload)
    except Exception:
        log.exception("Failed to submit evaluation payload to the exported")

    # # Debug evaluation log
    # ev_data = sorted([f"{k}={v!r}" for k, v in ev.model_dump().items() if v is not None])
    # get_logger().warning(f"Evaluation({', '.join(ev_data)})")


def coerce_eval_output_type(ev_output: typing.Any, class_name: Optional[str], func_name: str) -> EvaluationResult:
    if isinstance(ev_output, EvaluationResult):
        return ev_output
    if isinstance(ev_output, bool):
        return EvaluationResult(pass_=ev_output, score=float(ev_output))
    if isinstance(ev_output, (int, float)):
        return EvaluationResult(score=float(ev_output))
    if isinstance(ev_output, str):
        return EvaluationResult(text_output=ev_output)

    e_name = func_name
    if class_name:
        e_name = f"{class_name}.{func_name}"

    raise TypeError(
        f"Evaluator function '{e_name}' returned unexpected type {type(ev_output)!r}. "
        f"Supported return types are EvaluationResult, int, float, bool, str. "
    )


def evaluator(
    *,
    # Name for the evaluator. Defaults to function name (or class name in case of class based evaluators).
    evaluator_id: Optional[str] = None,
    # Name for the evaluation metric. Defaults to evaluator_id value.
    metric_name: Optional[str] = None,
    **kwargs,
):
    def decorator(fn):
        fn_sign = inspect.signature(fn)
        class_name: Optional[str] = kwargs.get("class_name")
        fn_name: str = fn.__name__
        eval_id: str = evaluator_id or fn_name
        met_name: str = metric_name or eval_id
        met_description: str = inspect.getdoc(fn)

        @traced(ignore_input=True, ignore_output=True)
        @functools.wraps(fn)
        async def wrapper_async(*fn_args, **fn_kwargs):
            start = time.perf_counter()
            try:
                ret = await fn(*fn_args, **fn_kwargs)
            except Exception as e:
                get_logger().exception("Evaluation failed")
                raise e
            elapsed = time.perf_counter() - start
            handle_eval_output(
                class_name=class_name,
                fn_name=fn_name,
                evaluator_id=eval_id,
                metric_name=met_name,
                metric_description=met_description,
                fn_sign=fn_sign,
                args=fn_args,
                kwargs=fn_kwargs,
                ret_value=ret,
                duration=datetime.timedelta(seconds=elapsed),
            )
            return ret

        @traced(ignore_input=True, ignore_output=True)
        @functools.wraps(fn)
        def wrapper_sync(*fn_args, **fn_kwargs):
            start = time.perf_counter()
            try:
                ret = fn(*fn_args, **fn_kwargs)
            except Exception as e:
                get_logger().exception("Evaluation failed")
                raise e
            elapsed = time.perf_counter() - start
            handle_eval_output(
                class_name=class_name,
                fn_name=fn_name,
                evaluator_id=eval_id,
                metric_name=met_name,
                metric_description=met_description,
                fn_sign=fn_sign,
                args=fn_args,
                kwargs=fn_kwargs,
                ret_value=ret,
                duration=datetime.timedelta(seconds=elapsed),
            )
            return ret

        if inspect.iscoroutinefunction(fn):
            return wrapper_async
        else:
            return wrapper_sync

    return decorator


# Inherit from ABCMeta so we can enforce abstractmethods
class _EvaluatorMeta(abc.ABCMeta):
    def __new__(mcls, name, bases, namespace, /, **kwargs):
        # Wrap evaluate method with evaluator decorator as soon as class is created.
        # This happens on class creation, not class instantiation.
        if "evaluate" in namespace:
            evaluator_id = namespace.get("name", name)
            into_evaluator = evaluator(evaluator_id=evaluator_id)
            namespace["evaluate"] = into_evaluator(namespace["evaluate"])
        return super().__new__(mcls, name, bases, namespace, **kwargs)


class Evaluator(metaclass=_EvaluatorMeta):
    name = None

    @typing.overload
    def evaluate(self) -> Optional[EvaluationResult]:
        """
        Synchronous version of evaluate method.
        When inheriting directly from Evaluator class it's permitted to change parameters signature.
        Return type should stay unchanged.
        """

    @abc.abstractmethod
    async def evaluate(self) -> Optional[EvaluationResult]:
        """
        Asynchronous version of evaluate method.
        When inheriting directly from Evaluator class it's permitted to change parameters signature.
        Return type should stay unchanged.
        """


class StructuredEvaluator(Evaluator):
    @typing.overload
    def evaluate(
        self,
        *,
        system_prompt: Optional[str] = None,
        task_context: Union[list[str], str, None] = None,
        task_input: Optional[str] = None,
        task_output: Optional[str] = None,
        gold_answer: Optional[str] = None,
        task_metadata: Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: Any,
    ) -> EvaluationResult: ...

    @abc.abstractmethod
    async def evaluate(
        self,
        *,
        system_prompt: Optional[str] = None,
        task_context: Union[list[str], str, None] = None,
        task_input: Optional[str] = None,
        task_output: Optional[str] = None,
        gold_answer: Optional[str] = None,
        task_metadata: Optional[typing.Dict[str, typing.Any]] = None,
        **kwargs: Any,
    ) -> EvaluationResult: ...


# TODO
# class PatronusRemoteEvaluator(StructuredEvaluator): ...
