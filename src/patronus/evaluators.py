import abc
import asyncio
import inspect
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

import typing_extensions

from . import api_types, types
from .async_utils import run_as_coro
from .datasets import Row
from .types import _EvalParent

EVALUATION_ARGS = [
    "row",
    "task_result",
    "evaluated_model_system_prompt",
    "evaluated_model_retrieved_context",
    "evaluated_model_input",
    "evaluated_model_output",
    "evaluated_model_gold_answer",
    "evaluated_model_attachments",
    "tags",
    "parent",
    "experiment_id",
    "dataset_id",
    "dataset_sample_id",
]

EvalF = typing.TypeVar("EvalF", bound=typing.Callable[..., typing.Any])


class EvaluatorError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class Evaluator(abc.ABC):
    name: str = ""
    criteria: Optional[str] = None
    accepted_args: set[str]
    remote_capture = False

    def __init__(
        self,
        accepted_args: Optional[set[str]] = None,
        *,
        evaluator_name: Optional[str] = None,
        criteria: Optional[str] = None,
    ):
        self.name = self.name or evaluator_name or self.__class__.__name__
        self.criteria = self.criteria or criteria
        self.accepted_args = accepted_args
        if not self.accepted_args:
            sig = inspect.signature(self.evaluate)
            param_keys = sig.parameters.keys()
            for name in param_keys:
                if name not in EVALUATION_ARGS:
                    raise ValueError(
                        f"{name!r} is not a valid evaluator argument. Valid arguments are: {EVALUATION_ARGS}"
                    )
            self.accepted_args = set(param_keys)

    def __repr__(self):
        return f"<Evaluator (id: {self.name!r}, criteria: {self.criteria!r}) of class {self.__class__.__name__}>"

    def display_name(self) -> str:
        if not self.criteria:
            return self.name
        return f"{self.name}:{self.criteria}"

    @property
    def column_prefix(self):
        """
        Returns a column prefix that is used in exported dataframe for result columns created by the evaluator.
        """
        p = self.criteria or ""
        return f"{self.name}.{p}"

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    async def execute(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        executor: ThreadPoolExecutor,
        row: Row,
        task_result: Optional[types.TaskResult],
        tags: dict[str, str],
        parent: _EvalParent,
        experiment_id: str,
        dataset_id: Optional[str] = None,
        dataset_sample_id: Optional[int] = None,
    ) -> Optional[types.EvaluatorOutput]:
        model_output = task_result and task_result.evaluated_model_output or row.evaluated_model_output
        kwargs = {
            "evaluated_model_system_prompt": row.evaluated_model_system_prompt,
            "evaluated_model_retrieved_context": row.evaluated_model_retrieved_context,
            "evaluated_model_input": row.evaluated_model_input,
            "evaluated_model_output": model_output,
            "evaluated_model_gold_answer": row.evaluated_model_gold_answer,
            "evaluated_model_attachments": row.evaluated_model_attachments,
            "experiment_id": experiment_id,
            "row": row,
            "task_result": task_result,
            "parent": parent,
            "dataset_id": dataset_id,
            "dataset_sample_id": dataset_sample_id,
            "tags": tags,
        }
        pass_kwargs = {k: v for k, v in kwargs.items() if k in self.accepted_args}
        start_time = time.perf_counter()
        result = await run_as_coro(
            __loop=loop,
            __executor=executor,
            __fn=self.evaluate,
            **pass_kwargs,
        )
        if result is None:
            return None

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        if isinstance(result, bool):
            result = types.EvaluationResult(pass_=result, score_raw=float(result))
        if isinstance(result, (int, float)):
            result = types.EvaluationResult(pass_=None, score_raw=float(result))
        if isinstance(result, api_types.EvaluationResult):
            elapsed = result.evaluation_duration and result.evaluation_duration.seconds
        if isinstance(result, types.EvaluationResult):
            if not result.evaluation_duration_s:
                result.evaluation_duration_s = elapsed
        if not isinstance(result, (types.EvaluationResult, api_types.EvaluationResult)):
            raise ValueError(
                f"Evaluator {self.display_name()!r} returned an object of unexpected type {type(result)!r}"
            )

        return types.EvaluatorOutput(result=result, duration=elapsed)

    @abc.abstractmethod
    def evaluate(self, **kwargs): ...


class SyncFunctionalEvaluator(Evaluator):
    fn: EvalF

    def __init__(self, *, evaluator_name: str, criteria: Optional[str] = None, fn: EvalF, accepted_args: set[str]):
        self.fn = fn
        super().__init__(accepted_args, evaluator_name=evaluator_name, criteria=criteria)

    def evaluate(self, **kwargs):
        return self.fn(**kwargs)


class FunctionalEvaluator(Evaluator):
    fn: EvalF

    def __init__(self, *, evaluator_name: str, criteria: Optional[str] = None, fn: EvalF, accepted_args: set[str]):
        self.fn = fn
        super().__init__(accepted_args, evaluator_name=evaluator_name, criteria=criteria)

    async def evaluate(self, **kwargs):
        return await self.fn(**kwargs)


def evaluator(
    func=None, *, name: str = None, criteria: str = None
) -> Union[Evaluator, typing.Callable[[...], Evaluator]]:
    def decorator(fn: EvalF) -> Evaluator:
        sig = inspect.signature(fn)
        param_keys = sig.parameters.keys()
        for param_name in param_keys:
            if param_name not in EVALUATION_ARGS:
                raise ValueError(
                    f"{param_name!r} is not a valid evaluator argument. Valid arguments are: {EVALUATION_ARGS}"
                )
        evaluator_name = name or fn.__name__
        if inspect.iscoroutinefunction(fn):
            return FunctionalEvaluator(
                evaluator_name=evaluator_name,
                criteria=criteria,
                fn=fn,
                accepted_args=set(param_keys),
            )
        else:
            return SyncFunctionalEvaluator(
                evaluator_name=evaluator_name,
                criteria=criteria,
                fn=fn,
                accepted_args=set(param_keys),
            )

    if func is None:
        return decorator
    return decorator(func)


@typing_extensions.deprecated(
    "'simple_evaluator' is deprecated and will be removed in future versions. Use '@evaluator' decorator instead."
)
def simple_evaluator(fn: typing.Callable[[str, str], bool], name: Optional[str] = None) -> FunctionalEvaluator:
    """
    Simple evaluator allows to wrap functions boolean evaluation:

    ```python
    exact_match = simple_evaluator(lambda output, gold_answer: output == gold_answer)
    ```

    It can be also used to decorate multi-line function.

    ```python
    @simple_evaluator
    def iexact_match(output, gold_answer) -> bool:
        output = output.strip().lower()
        gold_answer = gold_answer.strip().lower()
        return output == gold_answer
    ```
    """

    def wrapper(evaluated_model_output: str, evaluated_model_gold_answer: str):
        passed = fn(evaluated_model_output, evaluated_model_gold_answer)
        return types.EvaluationResult(pass_=passed, score_raw=float(passed))

    name = name or fn.__name__
    if name == "<lambda>":
        name = "simple_evaluator"
    return evaluator(wrapper, evaluator_name=name)
