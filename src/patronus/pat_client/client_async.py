from asyncio import Task
from concurrent.futures import ThreadPoolExecutor

import asyncio
import collections
import inspect
import typing
from typing import Optional, Union, TypedDict, List

from .. import evals, EvaluationResult
from ..evals.evaluators import bundled_eval

_EvaluatorID = str
_Criteria = str


class EvaluatorDict(TypedDict, total=False):
    evaluator_id: str
    criteria: Optional[_Criteria]


Evaluator = Union[
    evals.StructuredEvaluator,
    evals.AsyncStructuredEvaluator,
    # TODO add support later
    # EvaluatorDict,
    # Tuple[_EvaluatorID, _Criteria],
    # _EvaluatorID
]


async def with_semaphore(sem: asyncio.Semaphore, coro: typing.Coroutine):
    async with sem:
        return await coro


class AsyncPatronus:
    def __init__(self, max_workers: int = 10):
        self._pending_tasks = collections.deque()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore = asyncio.Semaphore(max_workers)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def evaluate(
        self,
        evaluators: Union[List[Evaluator], Evaluator],
        *,
        system_prompt: Optional[str] = None,
        task_context: Union[list[str], str, None] = None,
        task_input: Optional[str] = None,
        task_output: Optional[str] = None,
        gold_answer: Optional[str] = None,
        task_metadata: Optional[dict] = None,
        return_exceptions: bool = False,
    ):
        singular_eval = not isinstance(evaluators, list)
        if singular_eval:
            evaluators = [evaluators]

        def into_coro(fn, **kwargs):
            if inspect.iscoroutinefunction(fn):
                coro = fn(**kwargs)
            else:
                coro = asyncio.to_thread(fn, **kwargs)
            return with_semaphore(self._semaphore, coro)

        with bundled_eval():
            return await asyncio.gather(
                *(
                    into_coro(
                        ev.evaluate,
                        system_prompt=system_prompt,
                        task_context=task_context,
                        task_input=task_input,
                        task_output=task_output,
                        gold_answer=gold_answer,
                        task_metadata=task_metadata,
                    )
                    for ev in evaluators
                ),
                return_exceptions=return_exceptions,
            )

    def evaluate_bg(
        self,
        evaluators: Union[List[Evaluator], Evaluator],
        *,
        system_prompt: Optional[str] = None,
        task_context: Union[list[str], str, None] = None,
        task_input: Optional[str] = None,
        task_output: Optional[str] = None,
        gold_answer: Optional[str] = None,
        task_metadata: Optional[dict] = None,
    ) -> Task[list[Union[EvaluationResult, Exception]]]:
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self.evaluate(
                evaluators=evaluators,
                system_prompt=system_prompt,
                task_context=task_context,
                task_input=task_input,
                task_output=task_output,
                gold_answer=gold_answer,
                task_metadata=task_metadata,
                return_exceptions=True,
            ),
            name="evaluate_bg",
        )
        self._pending_tasks.append(task)
        task.add_done_callback(self._consume_tasks)
        return task

    def _consume_tasks(self, task):
        while len(self._pending_tasks) > 0:
            task: Task = self._pending_tasks[0]
            if task.done():
                self._pending_tasks.popleft()
            else:
                return

    async def close(self):
        while len(self._pending_tasks) != 0:
            await self._pending_tasks.popleft()
