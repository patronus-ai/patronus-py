import decimal
import inspect

import typing

import asyncio

import abc
from decimal import Decimal

from typing import Union, Optional

from patronus import evals
from patronus import datasets
from patronus.evals.evaluators import coerce_eval_output_type
from patronus.experiments.types import TaskResult, EvalParent
from patronus.evals import EvaluationResult


class BaseEvaluatorAdapter(abc.ABC):
    """
    Abstract base class for all evaluator adapters.

    Evaluator adapters provide a standardized interface between the experiment framework
    and various types of evaluators (function-based, class-based, etc.).

    All concrete adapter implementations must inherit from this class and implement
    the required abstract methods.
    """

    _weight: Optional[Union[str, float]] = None

    @property
    @abc.abstractmethod
    def evaluator_id(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def criteria(self) -> Optional[str]:
        pass

    @property
    def weight(self) -> Optional[Union[str, float]]:
        return self._weight

    @property
    def canonical_name(self) -> str:
        return f"{self.evaluator_id}:{self.criteria or ''}"

    @abc.abstractmethod
    async def evaluate(
        self,
        row: datasets.Row,
        task_result: Optional[TaskResult],
        parent: EvalParent,
        **kwargs,
    ) -> EvaluationResult: ...


class EvaluatorAdapter(BaseEvaluatorAdapter):
    """
    Adapter for class-based evaluators conforming to the Evaluator or AsyncEvaluator protocol.

    This adapter enables the use of evaluator classes that implement either the Evaluator
    or AsyncEvaluator interface within the experiment framework.

    Attributes:
        evaluator: The evaluator instance to adapt.

    **Examples:**

    ```python
    import typing
    from typing import Optional

    from patronus import datasets
    from patronus.evals import Evaluator, EvaluationResult
    from patronus.experiments import run_experiment
    from patronus.experiments.adapters import EvaluatorAdapter
    from patronus.experiments.types import TaskResult, EvalParent


    class MatchEvaluator(Evaluator):
        def __init__(self, sanitizer=None):
            if sanitizer is None:
                sanitizer = lambda x: x
            self.sanitizer = sanitizer

        def evaluate(self, actual: str, expected: str) -> EvaluationResult:
            matched = self.sanitizer(actual) == self.sanitizer(expected)
            return EvaluationResult(pass_=matched, score=int(matched))


    exact_match = MatchEvaluator()
    fuzzy_match = MatchEvaluator(lambda x: x.strip().lower())


    class MatchAdapter(EvaluatorAdapter):
        def __init__(self, evaluator: MatchEvaluator):
            super().__init__(evaluator)

        def transform(
            self,
            row: datasets.Row,
            task_result: Optional[TaskResult],
            parent: EvalParent,
            **kwargs
        ) -> tuple[list[typing.Any], dict[str, typing.Any]]:
            args = [row.task_output, row.gold_answer]
            kwargs = {}
            # Passing arguments via kwargs would also work in this case.
            # kwargs = {"actual": row.task_output, "expected": row.gold_answer}
            return args, kwargs


    run_experiment(
        dataset=[{"task_output": "string\t", "gold_answer": "string"}],
        evaluators=[MatchAdapter(exact_match), MatchAdapter(fuzzy_match)],
    )
    ```

    """

    evaluator: Union[evals.Evaluator, evals.AsyncEvaluator]

    def __init__(self, evaluator: evals.Evaluator):
        if not isinstance(evaluator, evals.Evaluator):
            raise TypeError(f"{evaluator} is not {evals.Evaluator.__name__}.")
        self.evaluator = evaluator

    @property
    def weight(self) -> Optional[Union[str, float]]:
        return self.evaluator.weight

    @property
    def evaluator_id(self) -> str:
        return self.evaluator.get_evaluator_id()

    @property
    def criteria(self) -> Optional[str]:
        return self.evaluator.get_criteria()

    def transform(
        self,
        row: datasets.Row,
        task_result: Optional[TaskResult],
        parent: EvalParent,
        **kwargs: typing.Any,
    ) -> tuple[list[typing.Any], dict[str, typing.Any]]:
        """
        Transform experiment framework arguments to evaluation method arguments.

        Args:
            row: The data row being evaluated.
            task_result: The result of the task execution, if available.
            parent: The parent evaluation context.
            **kwargs: Additional keyword arguments from the experiment.

        Returns:
            A list of positional arguments to pass to the evaluator function.
            A dictionary of keyword arguments to pass to the evaluator function.
        """

        return (
            [],
            {"row": row, "task_result": task_result, "parent": parent, **kwargs},
        )

    async def evaluate(
        self,
        row: datasets.Row,
        task_result: Optional[TaskResult],
        parent: EvalParent,
        **kwargs: typing.Any,
    ) -> EvaluationResult:
        """
        Evaluate the given row and task result using the adapted evaluator function.

        This method implements the BaseEvaluatorAdapter.evaluate() protocol.

        Args:
            row: The data row being evaluated.
            task_result: The result of the task execution, if available.
            parent: The parent evaluation context.
            **kwargs: Additional keyword arguments from the experiment.

        Returns:
            An EvaluationResult containing the evaluation outcome.
        """
        ev_args, ev_kwargs = self.transform(row, task_result, parent, **kwargs)
        return await self._evaluate(*ev_args, **ev_kwargs)

    async def _evaluate(self, *args: typing.Any, **kwargs: typing.Any) -> TaskResult:
        if isinstance(self.evaluator, evals.AsyncEvaluator):
            return await self.evaluator.evaluate(*args, **kwargs)
        elif isinstance(self.evaluator, evals.Evaluator):
            return await asyncio.to_thread(self.evaluator.evaluate, *args, **kwargs)
        else:
            raise RuntimeError("Invalid evaluator type")


class StructuredEvaluatorAdapter(EvaluatorAdapter):
    """
    Adapter for structured evaluators.
    """

    evaluator: Union[evals.StructuredEvaluator, evals.AsyncStructuredEvaluator]

    def __init__(
        self,
        evaluator: Union[evals.StructuredEvaluator, evals.AsyncStructuredEvaluator],
    ):
        if not isinstance(evaluator, (evals.StructuredEvaluator, evals.AsyncStructuredEvaluator)):
            raise TypeError(
                f"{type(evaluator)} is not "
                f"{evals.AsyncStructuredEvaluator.__name__} nor {evals.StructuredEvaluator.__name__}."
            )
        super().__init__(evaluator)

    def transform(
        self,
        row: datasets.Row,
        task_result: Optional[TaskResult],
        parent: EvalParent,
        **kwargs: typing.Any,
    ) -> tuple[list[typing.Any], dict[str, typing.Any]]:
        task_output = row.task_output
        task_metadata = row.task_metadata
        if task_result is not None:
            task_output = task_result.output
            task_metadata = task_result.metadata

        ev_kwargs = dict(
            system_prompt=row.system_prompt,
            task_context=row.task_context,
            task_attachments=row.task_attachments,
            task_input=row.task_input,
            task_output=task_output,
            gold_answer=row.gold_answer,
            task_metadata=task_metadata,
        )
        return [], ev_kwargs


class FuncEvaluatorAdapter(BaseEvaluatorAdapter):
    """
    Adapter class that allows using function-based evaluators with the experiment framework.

    This adapter serves as a bridge between function-based evaluators decorated with `@evaluator()`
    and the experiment framework's evaluation system.
    It handles both synchronous and asynchronous evaluator functions.

    Attributes:
        fn (Callable): The evaluator function to be adapted.

    Notes:
        - The function passed to this adapter must be decorated with `@evaluator()`.
        - The adapter automatically handles the conversion between function results and proper
          evaluation result objects.

    Examples:

        Direct usage with a compatible evaluator function:

        ```python
        from patronus import evaluator
        from patronus.experiments import FuncEvaluatorAdapter, run_experiment
        from patronus.datasets import Row


        @evaluator()
        def exact_match(row: Row, **kwargs):
            return row.task_output == row.gold_answer

        run_experiment(
            dataset=[{"task_output": "string", "gold_answer": "string"}],
            evaluators=[FuncEvaluatorAdapter(exact_match)]
        )
        ```

        Customized usage by overriding the `transform()` method:

        ```python
        from typing import Optional
        import typing

        from patronus import evaluator, datasets
        from patronus.experiments import FuncEvaluatorAdapter, run_experiment
        from patronus.experiments.types import TaskResult, EvalParent


        @evaluator()
        def exact_match(actual, expected):
            return actual == expected


        class AdaptedExactMatch(FuncEvaluatorAdapter):
            def __init__(self):
                super().__init__(exact_match)

            def transform(
                self,
                row: datasets.Row,
                task_result: Optional[TaskResult],
                parent: EvalParent,
                **kwargs
            ) -> tuple[list[typing.Any], dict[str, typing.Any]]:
                args = [row.task_output, row.gold_answer]
                kwargs = {}

                # Alternative: passing arguments via kwargs instead of args
                # args = []
                # kwargs = {"actual": row.task_output, "expected": row.gold_answer}

                return args, kwargs


        run_experiment(
            dataset=[{"task_output": "string", "gold_answer": "string"}],
            evaluators=[AdaptedExactMatch()],
        )
        ```

    """

    def __init__(self, fn: typing.Callable[..., typing.Any], weight: Optional[Union[str, float]] = None):
        if not hasattr(fn, "_pat_evaluator"):
            raise ValueError(
                f"Passed function {fn.__qualname__} is not an evaluator. "
                "Hint: add @evaluator decorator to the function."
            )

        if weight is not None:
            try:
                Decimal(str(weight))
            except (decimal.InvalidOperation, ValueError, TypeError):
                raise TypeError(
                    f"{weight} is not a valid weight. Weight must be a valid decimal number (string or float)."
                )

        self.fn = fn
        self._weight = weight

    @property
    def evaluator_id(self) -> str:
        # @evaluator() wrapper sets that value
        return self.fn._pat_evaluator_id  # noqa

    @property
    def criteria(self) -> Optional[str]:
        # @evaluator() wrapper sets that value
        return self.fn._pat_criteria  # noqa

    def transform(
        self,
        row: datasets.Row,
        task_result: Optional[TaskResult],
        parent: EvalParent,
        **kwargs: typing.Any,
    ) -> tuple[list[typing.Any], dict[str, typing.Any]]:
        """
        Transform experiment framework parameters to evaluator function parameters.

        Args:
            row: The data row being evaluated.
            task_result: The result of the task execution, if available.
            parent: The parent evaluation context.
            **kwargs: Additional keyword arguments from the experiment.

        Returns:
            A list of positional arguments to pass to the evaluator function.
            A dictionary of keyword arguments to pass to the evaluator function.
        """

        return (
            [],
            {"row": row, "task_result": task_result, "parent": parent, **kwargs},
        )

    async def evaluate(
        self,
        row: datasets.Row,
        task_result: Optional[TaskResult],
        parent: EvalParent,
        **kwargs: typing.Any,
    ) -> EvaluationResult:
        """
        Evaluate the given row and task result using the adapted evaluator function.

        This method implements the BaseEvaluatorAdapter.evaluate() protocol.

        Args:
            row: The data row being evaluated.
            task_result: The result of the task execution, if available.
            parent: The parent evaluation context.
            **kwargs: Additional keyword arguments from the experiment.

        Returns:
            An EvaluationResult containing the evaluation outcome.
        """
        ev_args, ev_kwargs = self.transform(row, task_result, parent, **kwargs)
        return await self._evaluate(*ev_args, **ev_kwargs)

    async def _evaluate(self, *args, **kwargs) -> TaskResult:
        if inspect.iscoroutinefunction(self.fn):
            result = await self.fn(*args, **kwargs)
        else:
            result = await asyncio.to_thread(self.fn, *args, **kwargs)

        return coerce_eval_output_type(result, self.fn.__qualname__)
