import abc
import asyncio
import inspect
import typing
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

import typing_extensions

from . import types
from .async_utils import run_as_coro
from .datasets import Row
from .types import _EvalParent

TASK_ARGS = {
    "row",
    "evaluated_model_system_prompt",
    "evaluated_model_retrieved_context",
    "evaluated_model_input",
    "evaluated_model_output",
    "evaluated_model_gold_answer",
    "tags",
    "parent",
}

TaskRetT = Union[types.TaskResult, str]
TaskFn = typing.Callable[..., Union[TaskRetT, typing.Awaitable[TaskRetT]]]


class Task(abc.ABC):
    name: str = ""

    def __init__(self, *, name: str, accepted_args: set[str]):
        self.name = self.name or name or self.__class__.__name__
        self.accepted_args = accepted_args

    def __repr__(self):
        return f"<Task with name {self.name!r} of class {self.__class__.__name__}>"

    async def execute(
        self,
        loop: asyncio.AbstractEventLoop,
        executor: ThreadPoolExecutor,
        row: Row,
        tags: dict[str, str],
        parent: _EvalParent,
    ) -> Optional[types.TaskResult]:
        kwargs = {
            "row": row,
            "evaluated_model_system_prompt": row.evaluated_model_system_prompt,
            "evaluated_model_retrieved_context": row.evaluated_model_retrieved_context,
            "evaluated_model_input": row.evaluated_model_input,
            "evaluated_model_output": row.evaluated_model_output,
            "evaluated_model_gold_answer": row.evaluated_model_gold_answer,
            "tags": {**tags},
            "parent": parent,
        }
        pass_kwargs = {k: v for k, v in kwargs.items() if k in self.accepted_args}
        result = await run_as_coro(
            __loop=loop,
            __executor=executor,
            __fn=self.task,
            **pass_kwargs,
        )
        if result is None:
            return None
        if isinstance(result, str):
            return types.TaskResult(evaluated_model_output=result)
        if isinstance(result, types.TaskResult):
            return result

        raise ValueError(f"Task {self.name!r} returned an object of unexpected type {type(result)!r}")

    @abc.abstractmethod
    def task(self, **kwargs) -> Union[TaskRetT, typing.Awaitable[TaskRetT]]: ...


class FunctionalTask(Task):
    fn: TaskFn

    def __init__(self, *, name: Optional[str] = None, fn: TaskFn, accepted_args: set[str]):
        self.fn = fn
        super().__init__(name=name or fn.__name__, accepted_args=accepted_args)

    async def task(self, **kwargs) -> TaskRetT:
        return await self.fn(**kwargs)


class SyncFunctionalTask(Task):
    fn: TaskFn

    def __init__(self, *, name: Optional[str] = None, fn: TaskFn, accepted_args: set[str]):
        self.fn = fn
        super().__init__(name=name or fn.__name__, accepted_args=accepted_args)

    def task(self, **kwargs) -> TaskRetT:
        return self.fn(**kwargs)


def task(func=None, *, name: Optional[str] = None) -> Union[Task, typing.Callable[[...], Task]]:
    def decorator(fn: TaskFn) -> Task:
        sig = inspect.signature(fn)
        param_keys = sig.parameters.keys()
        for param in param_keys:
            if param not in TASK_ARGS:
                raise ValueError(f"{param!r} is not a valid task argument. Valid arguments are: {TASK_ARGS}")
        if inspect.iscoroutinefunction(fn):
            return FunctionalTask(name=name, fn=fn, accepted_args=set(param_keys))
        else:
            return SyncFunctionalTask(name=name, fn=fn, accepted_args=set(param_keys))

    if func is None:
        return decorator
    return decorator(func)


@typing_extensions.deprecated(
    "'simple_task' is deprecated and will be removed in future versions. Use '@task' decorator instead."
)
def simple_task(lambda_fn: typing.Callable[[str], str]) -> Task:
    @task
    def wrapper(evaluated_model_input: str) -> str:
        return lambda_fn(evaluated_model_input)

    return wrapper


@task
@typing_extensions.deprecated(
    "'nop_task' is deprecated and will be removed in future versions.",
)
def nop_task(evaluated_model_output: str) -> str:
    return evaluated_model_output
