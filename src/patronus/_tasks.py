import abc
import asyncio
import dataclasses
import inspect
import typing
from concurrent.futures import ThreadPoolExecutor

from ._async_utils import run_as_coro

TASK_ARGS = {
    "evaluated_model_system_prompt",
    "evaluated_model_retrieved_context",
    "evaluated_model_input",
    "evaluated_model_output",
    "evaluated_model_gold_answer",
    "tags",
}


class TaskResultT(typing.Protocol):
    evaluated_model_output: str
    evaluated_model_system_prompt: str | None
    evaluated_model_name: str | None
    evaluated_model_provider: str | None
    evaluated_model_params: str | None
    evaluated_model_selected_model: str | None
    tags: dict[str, str] | None


@dataclasses.dataclass
class TaskResult:
    evaluated_model_output: str
    evaluated_model_system_prompt: str | None = None
    evaluated_model_name: str | None = None
    evaluated_model_provider: str | None = None
    evaluated_model_params: dict[str, int | float | str] | None = None
    evaluated_model_selected_model: str | None = None
    tags: dict[str, str] | None = None


TaskFn = typing.Callable[..., TaskResultT | str]


class Task(abc.ABC):
    def __init__(self, accepted_args: set[str]):
        self.accepted_args = accepted_args

    async def execute(
        self,
        loop: asyncio.AbstractEventLoop,
        executor: ThreadPoolExecutor,
        evaluated_model_system_prompt: str | None,
        evaluated_model_retrieved_context: list[str] | None,
        evaluated_model_input: str | None,
        evaluated_model_output: str | None,
        evaluated_model_gold_answer: str | None,
        tags: dict[str, str] | None = None,
    ) -> TaskResult:
        kwargs = {
            "evaluated_model_system_prompt": evaluated_model_system_prompt,
            "evaluated_model_retrieved_context": evaluated_model_retrieved_context,
            "evaluated_model_input": evaluated_model_input,
            "evaluated_model_output": evaluated_model_output,
            "evaluated_model_gold_answer": evaluated_model_gold_answer,
            "tags": {**tags},
        }
        pass_kwargs = {k: v for k, v in kwargs.items() if k in self.accepted_args}
        result = await run_as_coro(
            __loop=loop,
            __executor=executor,
            __fn=self.task,
            **pass_kwargs,
        )
        if isinstance(result, str):
            return TaskResult(evaluated_model_output=result)
        return result

    @abc.abstractmethod
    def task(self, **kwargs) -> TaskResultT | str | typing.Awaitable[TaskResultT | str]: ...


class FunctionalTask(Task):
    fn: TaskFn

    def __init__(self, fn: TaskFn, accepted_args: set[str]):
        self.fn = fn
        super().__init__(accepted_args)

    def task(self, **kwargs) -> TaskResultT | str:
        return self.fn(**kwargs)


def task(fn: TaskFn) -> Task:
    sig = inspect.signature(fn)
    param_keys = sig.parameters.keys()
    for name in param_keys:
        if name not in TASK_ARGS:
            raise ValueError(f"{name!r} is not a valid task argument. Valid arguments are: {TASK_ARGS}")
    return FunctionalTask(fn, set(param_keys))


def simple_task(lambda_fn: typing.Callable[[str], str]) -> Task:
    @task
    def wrapper(evaluated_model_input: str) -> str:
        return lambda_fn(evaluated_model_input)

    return wrapper


@task
def nop_task(evaluated_model_output: str) -> str:
    return evaluated_model_output
