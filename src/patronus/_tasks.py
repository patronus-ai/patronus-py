import abc
import dataclasses
import inspect
import typing

TASK_ARGS = {
    "evaluated_model_system_prompt",
    "evaluated_model_retrieved_context",
    "evaluated_model_input",
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


R = typing.TypeVar("R", bound=TaskResultT)
TaskFn = typing.Callable[..., R | str]


class Task(abc.ABC):
    def __init__(self, accepted_args: set[str]):
        self.accepted_args = accepted_args

    def __call__(
        self,
        evaluated_model_system_prompt: str | None,
        evaluated_model_retrieved_context: list[str] | None,
        evaluated_model_input: str | None,
        tags: dict[str, str] | None = None,
    ) -> R:
        kwargs = {
            "evaluated_model_system_prompt": evaluated_model_system_prompt,
            "evaluated_model_retrieved_context": evaluated_model_retrieved_context,
            "evaluated_model_input": evaluated_model_input,
            "tags": {**tags},
        }
        pass_kwargs = {k: v for k, v in kwargs.items() if k in self.accepted_args}
        result = self.execute(**pass_kwargs)
        if isinstance(result, str):
            return TaskResult(evaluated_model_output=result)
        return result

    @abc.abstractmethod
    def execute(self, **kwargs) -> str | R:
        ...


class FunctionalTask(Task):
    fn: TaskFn

    def __init__(self, fn: TaskFn, accepted_args: set[str]):
        self.fn = fn
        super().__init__(accepted_args)

    def execute(self, **kwargs) -> str | R:
        return self.fn(**kwargs)


def task(fn: TaskFn) -> Task:
    sig = inspect.signature(fn)
    param_keys = sig.parameters.keys()
    for name in param_keys:
        if name not in TASK_ARGS:
            raise ValueError(f"{name!r} is not a valid task argument. Valid arguments are: {TASK_ARGS}")
    return FunctionalTask(fn, set(param_keys))


def simple_task(lambda_fn):
    @task
    def wrapper(evaluated_model_input: str) -> str:
        return lambda_fn(evaluated_model_input)

    return wrapper
