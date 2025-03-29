import typing
from typing import Optional

from patronus.evals import EvaluationResult, Evaluator
from patronus.api import api_types

import pydantic


class TaskResult(pydantic.BaseModel):
    """
    Represents the result of a task with optional output, metadata, and tags.

    This class is used to encapsulate the result of a task, including optional
    fields for the output of the task, metadata related to the task, and any
    tags that can provide additional information or context about the task.

    Attributes:
        output: The output of the task, if any.
        metadata: Additional information or metadata associated with the task.
        tags: Key-value pairs used to tag and describe the task.
    """

    output: Optional[str] = None
    metadata: Optional[dict[str, typing.Any]] = None
    tags: Optional[dict[str, str]] = None


MaybeEvaluationResult = typing.Union[EvaluationResult, api_types.EvaluationResult, None]


class EvalsMap(dict):
    """
    A specialized dictionary for storing evaluation results with flexible key handling.

    This class extends dict to provide automatic key normalization for evaluation results,
    allowing lookup by evaluator objects, strings, or any object with a canonical_name attribute.
    """

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
    """
    Represents a node in the evaluation parent-child hierarchy, tracking task results and evaluations.

    Attributes:
        task: The task result associated with this evaluation node
        evals: A mapping of evaluator IDs to their evaluation results
        parent: Optional reference to a parent evaluation node, forming a linked list
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    task: Optional[TaskResult]
    evals: typing.Optional[EvalsMap]
    parent: typing.Optional["_EvalParent"]

    def find_eval_result(
        self, evaluator_or_name: typing.Union[str, Evaluator]
    ) -> typing.Union[api_types.EvaluationResult, EvaluationResult, None]:
        """
        Recursively searches for an evaluation result by evaluator ID or name.

        Args:
            evaluator_or_name: The evaluator ID, name, or object to search for

        Returns:
            The matching evaluation result, or None if not found
        """
        if not self.evals and self.parent:
            return self.parent.find_eval_result(evaluator_or_name)
        if evaluator_or_name in self.evals:
            return self.evals[evaluator_or_name]
        return None


_EvalParent.model_rebuild()


EvalParent = typing.Optional[_EvalParent]
"""
Type alias representing an optional reference to an evaluation parent,
used to track the hierarchy of evaluations and their results
"""
