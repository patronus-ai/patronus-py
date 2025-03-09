import asyncio
import httpx
import inspect
import os
import pandas as pd
import time
import typing
from typing import Optional, Any, Union

import typing_extensions as te

from patronus import context, datasets
from patronus.api import PatronusAPIClient, api_types
from patronus.context import get_tracer
from patronus.datasets import Dataset, DatasetLoader
from patronus.evals import StructuredEvaluator, AsyncStructuredEvaluator, bundled_eval, EvaluationResult
from patronus.evals.context import evaluation_attributes
from patronus.experiments.adapters import BaseEvaluatorAdapter, StructuredEvaluatorAdapter
from patronus.experiments.async_utils import run_until_complete
from patronus.experiments.reporter import Reporter
from patronus.experiments.tqdm import AsyncTQDMWithHandle
from patronus.experiments.types import EvalParent, TaskResult, _EvalParent, EvalsMap
from patronus.tracing import traced
from patronus.utils import merge_tags

# TODO Type hint them
Tags = dict[str, str]

T = typing.TypeVar("T")


class TaskProtocol(typing.Protocol[T]):
    def __call__(self, *, row: datasets.Row, parent: EvalParent, tags: Tags) -> T: ...


Task = Union[
    # Synchronous task signature
    TaskProtocol[Union[TaskResult, str, None]],
    # Asynchronous task signature
    TaskProtocol[typing.Awaitable[Union[TaskResult, str, None]]],
]

ExperimentDataset = Union[
    Dataset,
    DatasetLoader,
    list[dict[str, Any]],
    tuple[dict[str, Any], ...],
    pd.DataFrame,
    typing.Awaitable,
    typing.Callable[[], typing.Awaitable],
]

AdaptableEvaluators = Union[StructuredEvaluator, AsyncStructuredEvaluator, BaseEvaluatorAdapter]


class ChainLink(typing.TypedDict):
    task: Optional[Task]
    evaluators: list[AdaptableEvaluators]


class _ChainLink(typing.TypedDict):
    task: Optional[Task]
    evaluators: list[BaseEvaluatorAdapter]


def run_experiment(
    dataset: ExperimentDataset,
    task: Optional[Task] = None,
    evaluators: Optional[list[AdaptableEvaluators]] = None,
    chain: Optional[list[ChainLink]] = None,
    tags: Optional[Tags] = None,
    max_concurrency: int = 10,
    project_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> "Experiment":
    """
    Create and run an experiment.

    This function creates an experiment with the specified configuration and runs it to completion.
    The execution handling is context-aware:

    - When called from an asynchronous context (with a running event loop), it returns an
      awaitable that must be awaited.
    - When called from a synchronous context (no running event loop), it blocks until the
      experiment completes and returns the Experiment object.


    Returns
    -------
    Union[Experiment, Awaitable[Experiment]]
        In a synchronous context: the completed Experiment object.
        In an asynchronous context: an awaitable that resolves to the Experiment object.

    Examples:
        Synchronous execution:
            experiment = run_experiment(dataset, task=some_task)
            # Blocks until the experiment finishes.

        Asynchronous execution (e.g., in a Jupyter Notebook):
            experiment = await run_experiment(dataset, task=some_task)
            # Must be awaited within an async function or event loop.

    Notes:
        For manual control of the event loop, you can create and run the experiment as follows:
            experiment = await Experiment.create(...)
            await experiment.run()

    """

    async def _run_experiment() -> Union[Experiment, typing.Awaitable[Experiment]]:
        ex = await Experiment.create(
            dataset=dataset,
            task=task,
            evaluators=evaluators,
            chain=chain,
            tags=tags,
            max_concurrency=max_concurrency,
            project_name=project_name,
            experiment_name=experiment_name,
            api_key=api_key,
            **kwargs,
        )
        return await ex.run()

    return run_until_complete(_run_experiment())


class Experiment:
    project: Optional[api_types.Project]
    experiment: Optional[api_types.Experiment]
    tags: dict[str, str]
    # dataset is transformed raw dataset that is used by the experiment.
    dataset: Optional[Dataset]

    _project_name: Optional[str]
    _experiment_name: Optional[str]
    # _raw_dataset is a raw object passed to the constructor. It may be unset after Experiment is prepared.
    _raw_dataset: Optional[ExperimentDataset]

    _chain: list[_ChainLink]
    _started: bool

    _sem_tasks: asyncio.Semaphore
    _sem_evals: asyncio.Semaphore

    _api_key: Optional[str]

    _ctx: Optional[context.PatronusContext] = None

    def __init__(
        self,
        *,
        dataset: typing.Any,
        task: Optional[Task] = None,
        evaluators: Optional[list[AdaptableEvaluators]] = None,
        chain: Optional[list[ChainLink]] = None,
        tags: Optional[dict[str, str]] = None,
        max_concurrency: int = 10,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        # TODO
        # ...
        api_key: Optional[str] = None,
        **kwargs,
    ):
        if chain and evaluators:
            raise ValueError("Cannot specify both chain and evaluators")

        self._raw_dataset = dataset

        if not chain:
            chain = [{"task": task, "evaluators": evaluators}]
        self._chain = [
            {"task": _trace_task(link["task"]), "evaluators": _adapt_evaluators(link["evaluators"])} for link in chain
        ]
        self._started = False
        self._finished = False

        self._project_name = project_name
        self.project = None

        self._experiment_name = experiment_name
        self.experiment = None

        self.tags = tags or {}

        self.max_concurrency = max_concurrency

        self._api_key = api_key
        self._prepared = False

        self.reporter = Reporter()

    @classmethod
    async def create(
        cls,
        dataset: ExperimentDataset,
        task: Optional[Task] = None,
        evaluators: Optional[list[AdaptableEvaluators]] = None,
        chain: Optional[list[ChainLink]] = None,
        tags: Optional[Tags] = None,
        max_concurrency: int = 10,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> te.Self:
        ex = cls(
            dataset=dataset,
            task=task,
            evaluators=evaluators,
            chain=chain,
            tags=tags,
            max_concurrency=max_concurrency,
            project_name=project_name,
            experiment_name=experiment_name,
            api_key=api_key,
            **kwargs,
        )
        ex._ctx = await ex._prepare()

        return ex

    async def run(self) -> te.Self:
        if self._started:
            raise RuntimeError("Experiment already started")
        if self._prepared is False:
            raise ValueError(
                "Experiment must be prepared before starting. "
                "Seems that Experiment was not created using Experiment.create() classmethod."
            )
        self._started = True

        with context._CTX_PAT.using(self._ctx):
            await self._run()
            self._finished = True
            self.reporter.summary()

        await asyncio.to_thread(self._ctx.exporter.force_flush)
        await asyncio.to_thread(self._ctx.tracer_provider.force_flush)

        return self

    def to_dataframe(self) -> pd.DataFrame:
        if self._finished is not True:
            raise RuntimeError("Experiment has to be in finished state")
        return self.reporter.to_dataframe()

    def to_csv(self, path_or_buf, **kwargs) -> Optional[str]:
        return self.to_dataframe().to_csv(path_or_buf, **kwargs)

    async def _prepare(self) -> context.PatronusContext:
        # Creating the semaphores here instead of in __init__ to make sure
        # we initialize them in context of an event loop that will run the experiment.
        self._sem_tasks = asyncio.Semaphore(self.max_concurrency)
        self._sem_evals = asyncio.Semaphore(self.max_concurrency)

        from patronus.config import config
        from patronus.init import build_context

        cfg = config()

        client_http = httpx.Client(timeout=cfg.timeout_s)
        client_http_async = httpx.AsyncClient(timeout=cfg.timeout_s)

        api = PatronusAPIClient(
            client_http_async=client_http_async,
            client_http=client_http,
            base_url=cfg.api_url,
            api_key=self._api_key or cfg.api_key,
        )

        dataset = await self._prepare_dataset(self._raw_dataset)
        self._raw_dataset = None
        self.dataset = dataset

        self.project = await self._get_or_create_project(api, self._project_name or cfg.project_name)
        self._project_name = None

        self.experiment = await self._create_experiment(api, self.project.id, self._experiment_name, self.tags)
        self._experiment_name = None

        ctx = build_context(
            project_name=self.project.name,
            app=None,
            experiment_id=self.experiment.id,
            api_url=cfg.api_url,
            otel_endpoint=cfg.otel_endpoint,
            api_key=self._api_key or cfg.api_key,
            client_http=client_http,
            client_http_async=client_http_async,
        )
        self._prepared = True
        return ctx

    async def _run(self):
        title = f"Experiment  {self.project.name}/{self.experiment.name}"
        print("=" * len(title))

        tasks = [
            with_semaphore(self._sem_tasks, self._run_chain(idx, row))
            for idx, row in enumerate(self.dataset.iterrows(), start=1)
        ]

        tqdm = await AsyncTQDMWithHandle.prep_gather(*tasks, desc=title, unit="sample")
        self.reporter.set_tqdm(tqdm)
        await tqdm.gather()

    @classmethod
    async def _prepare_dataset(cls, dataset: Any) -> Dataset:
        if isinstance(dataset, Dataset):
            return dataset
        elif isinstance(dataset, DatasetLoader):
            return await dataset.load()
        elif isinstance(dataset, (list, tuple)):
            return Dataset.from_records(dataset)
        elif inspect.iscoroutine(dataset):
            return await cls._prepare_dataset(await dataset)
        elif inspect.iscoroutinefunction(dataset):
            return await cls._prepare_dataset(await dataset())
        elif callable(dataset):
            return await cls._prepare_dataset(dataset())
        elif isinstance(dataset, pd.DataFrame):
            return Dataset.from_dataframe(dataset)
        else:
            raise ValueError(f"'dataset' passed to the experiment is an unexpected object of type {type(dataset)!r}")

    @staticmethod
    async def _get_or_create_project(api: PatronusAPIClient, project_name: str) -> api_types.Project:
        return await api.create_project(api_types.CreateProjectRequest(name=project_name))

    @staticmethod
    async def _create_experiment(
        api: PatronusAPIClient, project_id: str, experiment_name: str, tags: Tags
    ) -> api_types.Experiment:
        name = generate_experiment_name(experiment_name)
        return await api.create_experiment(
            api_types.CreateExperimentRequest(
                project_id=project_id,
                name=name,
                tags=tags,
            )
        )

    async def _run_chain(self, idx: int, row: datasets.Row):
        tracer = get_tracer()
        span_name = f"Sample {idx}/{self.dataset.df.shape[0]}"
        with tracer.start_as_current_span(span_name):
            parent = None

            for link_idx, eval_link in enumerate(self._chain):
                # TODO come up with better span name
                with tracer.start_as_current_span(f"Stage {link_idx+1}"):
                    task = eval_link["task"]
                    adapted_evaluators: list[BaseEvaluatorAdapter] = eval_link["evaluators"]

                    outgoing_tags = merge_tags({}, row.tags or {}, experiment_tags=self.tags)
                    task_result: Optional[TaskResult] = None

                    if task is not None:
                        try:
                            task_result = await self.execute_task(task, row, parent, outgoing_tags)
                        except Exception as exc:
                            self.reporter.add_task_error(exc, row)
                            return

                        # If task returned None it means the record processing should be skipped
                        if task_result is None:
                            return

                    if task_result is not None and task_result.tags:
                        outgoing_tags = merge_tags(outgoing_tags, task_result.tags, experiment_tags=self.tags)

                    results = await self.evaluate_stage(adapted_evaluators, row, task_result, parent, outgoing_tags)

                    has_eval_errors = False
                    eval_results_map = EvalsMap()

                    for adapter, result in zip(adapted_evaluators, results):
                        if isinstance(result, Exception):
                            has_eval_errors = True
                            self.reporter.add_evaluator_error(result, row, adapter.evaluator_id, adapter.criteria)
                            continue

                        eval_results_map[adapter.canonical_name] = result

                        await self.reporter.add_result(
                            link_idx,
                            task.__name__ if task else None,
                            task_result,
                            adapter.evaluator_id,
                            adapter.criteria,
                            result,
                            row,
                        )

                    if has_eval_errors:
                        return

                    parent = _EvalParent(task=task_result, evals=eval_results_map, parent=parent)

    async def execute_task(self, task, row: datasets.Row, parent: EvalParent, tags: Tags) -> Optional[TaskResult]:
        try:
            if inspect.iscoroutinefunction(task):
                task_result = await task(row=row, parent=parent, tags=tags)
            else:
                # TODO handle with thread pool executor
                task_result = await asyncio.to_thread(task, row=row, parent=parent, tags=tags)
        except TypeError as e:
            error_msg = str(e)
            if "got an unexpected keyword argument" in error_msg:
                raise TypeError(
                    f"{error_msg}\n\nHint: You may need to update your task function signature. "
                    f"Either add the missing parameter to your function definition, or use "
                    f"**kwargs to accept any additional parameters."
                ) from e
            raise

        if task_result is None:
            return None

        if isinstance(task_result, TaskResult):
            return task_result

        if isinstance(task_result, str):
            return TaskResult(output=task_result, metadata=None, tags=None)

        raise TypeError(
            f"task returned unexpected unexpected type {type(task_result)!r}. "
            f"Allowed types: {TaskResult.__name__!r}, 'str' and 'NoneType'."
        )

    async def evaluate_stage(
        self,
        adapted_evaluators: list[BaseEvaluatorAdapter],
        row: datasets.Row,
        task_result: TaskResult,
        parent: EvalParent,
        tags: Tags,
    ) -> list[EvaluationResult]:
        attrs = {"tags": tags, "experiment_tags": self.tags, "dataset_id": row.dataset_id, "dataset_sample_id": row.sid}
        with evaluation_attributes(attrs=attrs):
            evals_gen = (
                with_semaphore(self._sem_evals, adapter.evaluate(row, task_result, parent))
                for adapter in adapted_evaluators
            )

            if len(adapted_evaluators) > 1:
                with bundled_eval("Evaluations"):
                    results = await asyncio.gather(*evals_gen, return_exceptions=True)
            else:
                results = await asyncio.gather(*evals_gen, return_exceptions=True)
        return results


async def with_semaphore(sem, coro):
    async with sem:
        return await coro


def generate_experiment_name(name: str) -> str:
    ts = int(time.time())
    if name:
        return f"{name}-{ts}"
    try:
        login = os.getlogin()
        return f"{login}-{ts}"
    except OSError:  # Possible in-cluster error: No such device or address
        return str(ts)


def _adapt_evaluators(evaluators: list[AdaptableEvaluators]) -> list[BaseEvaluatorAdapter]:
    def into(e):
        if isinstance(e, BaseEvaluatorAdapter):
            return e
        return StructuredEvaluatorAdapter(e)

    return [into(e) for e in evaluators]


def _trace_task(task):
    if task is None:
        return None
    if hasattr(task, "_pat_traced"):
        return task
    return traced(attributes={"gen_ai.operation.name": "task"})(task)
