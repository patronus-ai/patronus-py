import typing
from typing import Optional

from patronus.evals import EvaluationResult
from patronus.api import api_types

import pydantic


class TaskResult(pydantic.BaseModel):
    output: Optional[str] = None
    metadata: Optional[dict[str, typing.Any]] = None
    tags: Optional[dict[str, str]] = None


MaybeEvaluationResult = typing.Union[EvaluationResult, api_types.EvaluationResult, None]


class EvalsMap(dict):
    def __contains__(self, item) -> bool:
        item = self._key(item)
        return super().__contains__(item)

    def __getitem__(self, item) -> MaybeEvaluationResult:
        item = self._key(item)
        return super().__getitem__(item)

    def __setitem__(self, key: str, value: MaybeEvaluationResult):
        key = self._key(key)
        return super().__setitem__(key, value)

    @staticmethod
    def _key(item):
        if isinstance(item, str):
            return item
        if hasattr(item, "canonical_name"):
            return item.canonical_name
        return item


class _EvalParent(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    task: Optional[TaskResult]
    evals: typing.Optional[EvalsMap]
    parent: typing.Optional["_EvalParent"]

    def find_eval_result(self, evaluator_or_name) -> typing.Union[api_types.EvaluationResult, EvaluationResult, None]:
        if not self.evals and self.parent:
            return self.parent.find_eval_result(evaluator_or_name)
        if evaluator_or_name in self.evals:
            return self.evals[evaluator_or_name]
        return None


_EvalParent.model_rebuild()

EvalParent = typing.Optional[_EvalParent]
