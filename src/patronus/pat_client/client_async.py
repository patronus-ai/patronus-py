from asyncio import Task
from concurrent.futures import ThreadPoolExecutor

import asyncio
import collections
import inspect
import typing
from typing import Optional, Union, TypedDict, List

from patronus.evals import bundled_eval
from patronus.evals import StructuredEvaluator
from patronus.evals import AsyncStructuredEvaluator
from patronus.evals import AsyncRemoteEvaluator
from .container import EvaluationContainer

_EvaluatorID = str
_Criteria = str


class EvaluatorDict(TypedDict, total=False):
    evaluator_id: str
    criteria: Optional[_Criteria]


Evaluator = Union[
    StructuredEvaluator,
    AsyncStructuredEvaluator,
    EvaluatorDict,
    tuple[_EvaluatorID, _Criteria],
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

    def _map_evaluators(self, evs: list[Evaluator]):
        def _into(ev: Evaluator):
            if isinstance(ev, tuple):
                return AsyncRemoteEvaluator(ev[0], ev[1])
            if isinstance(ev, dict):
                return AsyncRemoteEvaluator(ev["evaluator_id"], ev["criteria"])
            return ev

        return [_into(e) for e in evs]

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
    ) -> EvaluationContainer:
        """
        Run multiple evaluators in parallel.
        """
        singular_eval = not isinstance(evaluators, list)
        if singular_eval:
            evaluators = [evaluators]
        evaluators = self._map_evaluators(evaluators)

        def into_coro(fn, **kwargs):
            if inspect.iscoroutinefunction(fn):
                coro = fn(**kwargs)
            else:
                coro = asyncio.to_thread(fn, **kwargs)
            return with_semaphore(self._semaphore, coro)

        with bundled_eval():
            results = await asyncio.gather(
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
        return EvaluationContainer(results)

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
    ) -> Task[EvaluationContainer]:
        """
        Run multiple evaluators in parallel. The returned task will be a background task.
        """
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
        """
        Gracefully close the client. This will wait for all background tasks to finish.
        """
        while len(self._pending_tasks) != 0:
            await self._pending_tasks.popleft()
