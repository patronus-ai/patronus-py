import pydantic
import typing

from . import api_types


class TaskResult(pydantic.BaseModel):
    evaluated_model_output: str
    metadata: dict[str, typing.Any] | None = None
    tags: dict[str, str] | None = None


class EvaluationResult(pydantic.BaseModel):
    pass_: bool | None = None
    score_raw: float | None = None
    metadata: dict[str, typing.Any] | None = None
    tags: dict[str, str] | None = None


class EvaluatorOutput(pydantic.BaseModel):
    result: EvaluationResult | api_types.EvaluationResult
    duration: float


class EvalParent(pydantic.BaseModel):
    task: TaskResult | None
    evals: dict[str, EvaluationResult | api_types.EvaluationResult | None] | None
    parent: typing.Optional["EvalParent"]

    def fine_eval(self, name) -> api_types.EvaluationResult | EvaluationResult | None:
        if not self.evals and self.parent:
            return self.parent.fine_eval(name)
        if name in self.evals:
            return self.evals[name]
        return None


EvalParent.model_rebuild()
