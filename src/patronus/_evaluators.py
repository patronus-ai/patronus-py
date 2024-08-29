import asyncio
import dataclasses
import inspect
import abc
import time
import typing
from concurrent.futures import ThreadPoolExecutor

from ._async_utils import run_as_coro

EVALUATION_ARGS = [
    "experiment_id",
    "evaluated_model_system_prompt",
    "evaluated_model_retrieved_context",
    "evaluated_model_input",
    "evaluated_model_output",
    "evaluated_model_gold_answer",
    "dataset_id",
    "dataset_sample_id",
    "tags",
]

EvalF = typing.TypeVar("EvalF", bound=typing.Callable[..., typing.Any])


class EvaluationResultT(typing.Protocol):
    pass_: bool | None
    score_raw: float | None
    tags: dict[str, str] | None = None


@dataclasses.dataclass
class EvaluationResult:
    pass_: bool | None
    score_raw: float | None
    tags: dict[str, str] | None = None


@dataclasses.dataclass
class EvaluatorOutput:
    result: EvaluationResultT
    duration: float


# TODO we should have more classes of evaluator errors
#      some of them may be recoverable on retry e.g. on remote 500, 502, 429
#      and some not, e.g. ZeroDivisionError locally, or 400, 422 remote
class EvaluatorError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class Evaluator(abc.ABC):
    name = "unknown"
    profile_name = None
    accepted_args: set[str]
    remote_capture = False

    def __init__(
        self,
        accepted_args: set[str] | None = None,
        *,
        name: str | None = None,
    ):
        self.name = name or self.__class__.__name__
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

    def display_name(self) -> str:
        return self.name

    async def execute(
        self,
        loop: asyncio.AbstractEventLoop,
        executor: ThreadPoolExecutor,
        experiment_id: str,
        evaluated_model_system_prompt: str | None = None,
        evaluated_model_retrieved_context: list[str] | None = None,
        evaluated_model_input: str | None = None,
        evaluated_model_output: str | None = None,
        evaluated_model_gold_answer: str | None = None,
        dataset_id: str | None = None,
        dataset_sample_id: int | None = None,
        tags: dict[str, str] | None = None,
    ) -> EvaluatorOutput:
        kwargs = {
            "experiment_id": experiment_id,
            "evaluated_model_system_prompt": evaluated_model_system_prompt,
            "evaluated_model_retrieved_context": evaluated_model_retrieved_context,
            "evaluated_model_input": evaluated_model_input,
            "evaluated_model_output": evaluated_model_output,
            "evaluated_model_gold_answer": evaluated_model_gold_answer,
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
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        return EvaluatorOutput(result=result, duration=elapsed)

    @abc.abstractmethod
    def evaluate(self, **kwargs) -> EvaluationResultT | typing.Awaitable[EvaluationResultT]: ...


class FunctionalEvaluator(Evaluator):
    fn: EvalF

    def __init__(self, name: str, fn: EvalF, accepted_args: set[str]):
        self.name = name
        self.fn = fn
        super().__init__(accepted_args)

    def evaluate(self, **kwargs) -> EvaluationResultT:
        ret = self.fn(**kwargs)
        if isinstance(ret, bool):
            return EvaluationResult(pass_=ret, score_raw=float(ret))
        return ret


def evaluator(fn: EvalF, evaluator_name: str | None = None) -> FunctionalEvaluator:
    sig = inspect.signature(fn)
    param_keys = sig.parameters.keys()
    for name in param_keys:
        if name not in EVALUATION_ARGS:
            raise ValueError(f"{name!r} is not a valid evaluator argument. Valid arguments are: {EVALUATION_ARGS}")
    evaluator_name = evaluator_name or fn.__name__
    return FunctionalEvaluator(evaluator_name, fn, set(param_keys))


def simple_evaluator(fn: typing.Callable[[str, str], bool], name: str | None = None) -> FunctionalEvaluator:
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

    def wrapper(evaluated_model_output: str, evaluated_model_gold_answer: str) -> EvaluationResultT:
        passed = fn(evaluated_model_output, evaluated_model_gold_answer)
        return EvaluationResult(pass_=passed, score_raw=float(passed))

    name = name or fn.__name__
    if name == "<lambda>":
        name = "simple_evaluator"
    return evaluator(wrapper, evaluator_name=name)
