import atexit
import contextvars
import functools
import typing
from multiprocessing.pool import ThreadPool, AsyncResult, ApplyResult
from typing import Union, Optional

from .container import EvaluationContainer
from ..evals import StructuredEvaluator, EvaluationResult, RemoteEvaluator
from ..evals.evaluators import bundled_eval

T = typing.TypeVar("T")

_EvaluatorID = str
_Criteria = str


class EvaluatorDict(typing.TypedDict, total=False):
    evaluator_id: str
    criteria: Optional[_Criteria]


Evaluator = Union[
    StructuredEvaluator,
    EvaluatorDict,
    tuple[_EvaluatorID, _Criteria],
]


# Type Hint for multiprocessing.AsyncResult
# The stdlib doesn't provide a proper generic hint, so we create a monkey-patch-style type hint.
class TypedAsyncResult(typing.Generic[T]):
    def get(self, timeout: Optional[float] = None) -> T: ...
    def wait(self, timeout: Optional[float] = None) -> None: ...
    def ready(self) -> bool: ...
    def successful(self) -> bool: ...


def _into_thread_run_fn(eval_fn, *args, **kwargs) -> typing.Callable[[...], typing.Any]:
    # Prepare a function to run in a thread.
    # This function make sure that contextvars are propagated as it is necessary for proper
    # evaluation and telemetry tracing across thread boundary.
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, eval_fn, *args, **kwargs)
    return func_call


class Patronus:
    def __init__(self, workers: int = 10, shutdown_on_exit: bool = True):
        self._worker_pool = ThreadPool(workers)
        self._supervisor_pool = ThreadPool(workers)

        self._at_exit_handler = None
        if shutdown_on_exit:
            self._at_exit_handler = atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _submit_to_pool(self, callables) -> list[ApplyResult]:
        return [self._worker_pool.apply_async(func=cb) for cb in callables]

    def _process_batch(self, callables, *, return_exceptions: bool) -> list[Union[EvaluationResult, None, Exception]]:
        results = self._submit_to_pool(callables)

        def handle_result(res: AsyncResult):
            try:
                return res.get()
            except Exception as e:
                if return_exceptions:
                    return e
                else:
                    raise e

        return [handle_result(res) for res in results]

    def _map_evaluators(self, evs: list[Evaluator]):
        def _into(ev: Evaluator):
            if isinstance(ev, tuple):
                return RemoteEvaluator(ev[0], ev[1])
            if isinstance(ev, dict):
                return RemoteEvaluator(ev["evaluator_id"], ev["criteria"])
            return ev

        return [_into(e) for e in evs]

    def evaluate(
        self,
        evaluators: typing.Union[list[Evaluator], Evaluator],
        *,
        system_prompt: typing.Optional[str] = None,
        task_context: typing.Union[list[str], str, None] = None,
        task_input: typing.Optional[str] = None,
        task_output: typing.Optional[str] = None,
        gold_answer: typing.Optional[str] = None,
        task_metadata: typing.Optional[dict[str, typing.Any]] = None,
        return_exceptions: bool = False,
    ) -> EvaluationContainer:
        """
        Run multiple evaluators in parallel.
        """
        if not isinstance(evaluators, list):
            evaluators = [evaluators]
        evaluators = self._map_evaluators(evaluators)

        with bundled_eval():
            callables = [
                _into_thread_run_fn(
                    ev.evaluate,
                    system_prompt=system_prompt,
                    task_context=task_context,
                    task_input=task_input,
                    task_output=task_output,
                    gold_answer=gold_answer,
                    task_metadata=task_metadata,
                )
                for ev in evaluators
            ]
            results = self._process_batch(callables, return_exceptions=return_exceptions)
            return EvaluationContainer(results)

    def evaluate_bg(
        self,
        evaluators: list[StructuredEvaluator],
        *,
        system_prompt: typing.Optional[str] = None,
        task_context: typing.Union[list[str], str, None] = None,
        task_input: typing.Optional[str] = None,
        task_output: typing.Optional[str] = None,
        gold_answer: typing.Optional[str] = None,
        task_metadata: typing.Optional[dict[str, typing.Any]] = None,
    ) -> TypedAsyncResult[EvaluationContainer]:
        """
        Run multiple evaluators in parallel. The returned task will be a background task.
        """

        def _run():
            with bundled_eval():
                callables = [
                    _into_thread_run_fn(
                        ev.evaluate,
                        system_prompt=system_prompt,
                        task_context=task_context,
                        task_input=task_input,
                        task_output=task_output,
                        gold_answer=gold_answer,
                        task_metadata=task_metadata,
                    )
                    for ev in evaluators
                ]
                results = self._process_batch(callables, return_exceptions=True)
                return EvaluationContainer(results)

        return typing.cast(
            TypedAsyncResult[EvaluationContainer], self._supervisor_pool.apply_async(_into_thread_run_fn(_run))
        )

    def close(self):
        """
        Gracefully close the client. This will wait for all background tasks to finish.
        """
        self._close()
        if self._at_exit_handler:
            atexit.unregister(self._at_exit_handler)

    def _close(self):
        self._supervisor_pool.close()
        self._supervisor_pool.join()
        self._worker_pool.close()
        self._worker_pool.join()
