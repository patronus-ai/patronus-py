import pathlib

import contextlib

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
from patronus.evals import (
    AsyncRemoteEvaluator,
    StructuredEvaluator,
    AsyncStructuredEvaluator,
    bundled_eval,
    EvaluationResult,
)
from patronus.evals.evaluators import RemoteEvaluator
from patronus.evals.context import evaluation_attributes
from patronus.experiments.adapters import BaseEvaluatorAdapter, StructuredEvaluatorAdapter
from patronus.experiments.async_utils import run_until_complete
from patronus.experiments.reporter import Reporter
from patronus.experiments.tqdm import AsyncTQDMWithHandle
from patronus.experiments.types import EvalParent, TaskResult, _EvalParent, EvalsMap
from patronus.tracing import traced
from patronus.tracing.attributes import Attributes, SpanTypes
from patronus.utils import merge_tags

Tags = dict[str, str]
"""
Tags are key-value pairs applied to experiments, task results and evaluation results.
"""

T = typing.TypeVar("T")


class TaskProtocol(typing.Protocol[T]):
    """
    Defines an interface for a task.

    Task is a function that processes each dataset row and produces output for evaluation.
    """

    def __call__(self, *, row: datasets.Row, parent: EvalParent, tags: Tags) -> T:
        """
        Processes a dataset row, using the provided context to produce task output.

        Args:
            row: The dataset row to process.
            parent: Reference to the parent task's output and evaluation results.
            tags: Key-value pairs.

        Returns:
            Task output of type T or None to skip the row processing.

        Example:
            ```python
            def simple_task(row: datasets.Row, parent: EvalParent, tags: Tags) -> TaskResult:
                # Process input from the dataset row
                input_text = row.task_input

                # Generate output
                output = f"Processed: {input_text}"

                # Return result
                return TaskResult(
                    output=output,
                    metadata={"processing_time_ms": 42},
                    tags={"model": "example-model"}
                )
            ```
        """


Task = Union[
    # Synchronous task signature
    TaskProtocol[Union[TaskResult, str, None]],
    # Asynchronous task signature
    TaskProtocol[typing.Awaitable[Union[TaskResult, str, None]]],
]
"""
A function that processes each dataset row and produces output for evaluation.
"""

ExperimentDataset = Union[
    Dataset,
    DatasetLoader,
    list[dict[str, Any]],
    tuple[dict[str, Any], ...],
    pd.DataFrame,
    typing.Awaitable,
    typing.Callable[[], typing.Awaitable],
]
"""
Any object that would "resolve" into [Dataset][patronus.datasets.datasets.Row].
"""

AdaptableEvaluators = Union[StructuredEvaluator, AsyncStructuredEvaluator, BaseEvaluatorAdapter]


class ChainLink(typing.TypedDict):
    """
    Represents a single stage in an experiment's processing chain.

    Each ChainLink contains an optional task function that processes dataset rows
    and a list of evaluators that assess the task's output.

    Attributes:
        task: Function that processes a dataset row and produces output.
        evaluators: List of evaluators to assess the task's output.
    """

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
    service: Optional[str] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    otel_endpoint: Optional[str] = None,
    otel_exporter_otlp_protocol: Optional[str] = None,
    ui_url: Optional[str] = None,
    timeout_s: Optional[int] = None,
    integrations: Optional[list[typing.Any]] = None,
    **kwargs,
) -> Union["Experiment", typing.Awaitable["Experiment"]]:
    """
    Create and run an experiment.

    This function creates an experiment with the specified configuration and runs it to completion.
    The execution handling is context-aware:

    - When called from an asynchronous context (with a running event loop), it returns an
      awaitable that must be awaited.
    - When called from a synchronous context (no running event loop), it blocks until the
      experiment completes and returns the Experiment object.


    **Examples:**

    Synchronous execution:

    ```python
    experiment = run_experiment(dataset, task=some_task)
    # Blocks until the experiment finishes.
    ```

    Asynchronous execution (e.g., in a Jupyter Notebook):

    ```python
    experiment = await run_experiment(dataset, task=some_task)
    # Must be awaited within an async function or event loop.
    ```

    **Parameters:**

    See [Experiment.create][patronus.experiments.experiment.Experiment.create] for list of arguments.

    Returns:
        Experiment (Experiment): In a synchronous context: the completed Experiment object.
        Experiment (Awaitable[Experiment]): In an asynchronous context:
            an awaitable that resolves to the Experiment object.

    Notes:
        For manual control of the event loop, you can create and run the experiment as follows:

        ```python
        experiment = await Experiment.create(...)
        await experiment.run()
        ```

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
            service=service,
            api_key=api_key,
            api_url=api_url,
            otel_endpoint=otel_endpoint,
            otel_exporter_otlp_protocol=otel_exporter_otlp_protocol,
            ui_url=ui_url,
            timeout_s=timeout_s,
            integrations=integrations,
            **kwargs,
        )
        return await ex.run()

    return run_until_complete(_run_experiment())


class Experiment:
    """
    Manages evaluation experiments across datasets using tasks and evaluators.

    An experiment represents a complete evaluation pipeline that processes a dataset
    using defined tasks, applies evaluators to the outputs, and collects the results.
    Experiments track progress, create reports, and interface with the Patronus platform.

    Create experiment instances using the [`create()`][patronus.experiments.experiment.Experiment.create] class method
    or through the [`run_experiment()`][patronus.experiments.experiment.run_experiment] convenience function.
    """

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
    _prepared: bool

    _sem_tasks: asyncio.Semaphore
    _sem_evals: asyncio.Semaphore

    _service: Optional[str]
    _api_key: Optional[str]
    _api_url: Optional[str]
    _otel_endpoint: Optional[str]
    _otel_exporter_otlp_protocol: Optional[str]
    _ui_url: Optional[str]
    _timeout_s: Optional[int]

    _ctx: Optional[context.PatronusContext] = None

    def __init__(
        self,
        *,
        dataset: typing.Any,
        task: Optional[Task] = None,
        evaluators: Optional[list[AdaptableEvaluators]] = None,
        chain: Optional[list[ChainLink]] = None,
        tags: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        max_concurrency: int = 10,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        service: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        otel_endpoint: Optional[str] = None,
        otel_exporter_otlp_protocol: Optional[str] = None,
        ui_url: Optional[str] = None,
        timeout_s: Optional[int] = None,
        integrations: Optional[list[typing.Any]] = None,
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
        self.metadata = metadata

        self.max_concurrency = max_concurrency

        self._service = service
        self._api_key = api_key
        self._api_url = api_url
        self._otel_endpoint = otel_endpoint
        self._otel_exporter_otlp_protocol = otel_exporter_otlp_protocol
        self._ui_url = ui_url
        self._timeout_s = timeout_s

        self._prepared = False

        self.reporter = Reporter()

        self._integrations = integrations

    @classmethod
    async def create(
        cls,
        dataset: ExperimentDataset,
        task: Optional[Task] = None,
        evaluators: Optional[list[AdaptableEvaluators]] = None,
        chain: Optional[list[ChainLink]] = None,
        tags: Optional[Tags] = None,
        metadata: Optional[dict[str, Any]] = None,
        max_concurrency: int = 10,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        service: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        otel_endpoint: Optional[str] = None,
        otel_exporter_otlp_protocol: Optional[str] = None,
        ui_url: Optional[str] = None,
        timeout_s: Optional[int] = None,
        integrations: Optional[list[typing.Any]] = None,
        **kwargs: typing.Any,
    ) -> te.Self:
        """
        Creates an instance of the class asynchronously with the specified parameters while performing
        necessary preparations. This method initializes various attributes including dataset, task,
        evaluators, chain, and additional configurations for managing concurrency, project details,
        service information, API keys, timeout settings, and integrations.

        Use [run_experiment][patronus.experiments.experiment.run_experiment] for more convenient usage.

        Args:
            dataset: The dataset to run evaluations against.
            task: A function that processes each dataset row and produces output for evaluation.
                Mutually exclusive with the `chain` parameter.
            evaluators: A list of evaluators to assess the task output. Mutually exclusive with
                the `chain` parameter.
            chain: A list of processing stages, each containing a task and associated evaluators.
                Use this for multi-stage evaluation pipelines.
            tags: Key-value pairs.
                All evaluations created by the experiment will contain these tags.
            metadata: Arbitrary dict.
                Metadata associated with the experiment.
            max_concurrency: Maximum number of concurrent task and evaluation operations.
            project_name: Name of the project to create or use. Falls back to configuration or
                environment variables if not provided.
            experiment_name: Custom name for this experiment run. A timestamp will be appended.
            service: OpenTelemetry service name for tracing. Falls back to configuration or
                environment variables if not provided.
            api_key: API key for Patronus services. Falls back to configuration or environment
                variables if not provided.
            api_url: URL for the Patronus API. Falls back to configuration or environment
                variables if not provided.
            otel_endpoint: OpenTelemetry collector endpoint. Falls back to configuration or
                environment variables if not provided.
            otel_exporter_otlp_protocol: OpenTelemetry exporter protocol (grpc or http/protobuf).
                Falls back to configuration or environment variables if not provided.
            ui_url: URL for the Patronus UI. Falls back to configuration or environment
                variables if not provided.
            timeout_s: Timeout in seconds for API operations. Falls back to configuration or
                environment variables if not provided.
            integrations: A list of OpenTelemetry instrumentors for additional tracing capabilities.
            **kwargs: Additional keyword arguments passed to the experiment.

        Returns:
            Experiment: ...

        """
        ex = cls(
            dataset=dataset,
            task=task,
            evaluators=evaluators,
            chain=chain,
            tags=tags,
            metadata=metadata,
            max_concurrency=max_concurrency,
            project_name=project_name,
            experiment_name=experiment_name,
            service=service,
            api_key=api_key,
            api_url=api_url,
            otel_endpoint=otel_endpoint,
            otel_exporter_otlp_protocol=otel_exporter_otlp_protocol,
            ui_url=ui_url,
            timeout_s=timeout_s,
            integrations=integrations,
            **kwargs,
        )
        ex._ctx = await ex._prepare()

        return ex

    async def run(self) -> te.Self:
        """
        Executes the experiment by processing all dataset items.

        Runs the experiment's task chain on each dataset row, applying evaluators
        to the results and collecting metrics. Progress is displayed with a progress
        bar and results are logged to the Patronus platform.

        Returns:
            The experiment instance.
        """
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
        """
        Converts experiment results to a pandas DataFrame.

        Creates a tabular representation of all evaluation results with
        dataset identifiers, task information, evaluation scores, and metadata.

        Returns:
            A pandas DataFrame containing all experiment results.
        """
        if self._finished is not True:
            raise RuntimeError("Experiment has to be in finished state")
        return self.reporter.to_dataframe()

    def to_csv(
        self, path_or_buf: Union[str, pathlib.Path, typing.IO[typing.AnyStr]], **kwargs: typing.Any
    ) -> Optional[str]:
        """
        Saves experiment results to a CSV file.

        Converts experiment results to a DataFrame and saves them as a CSV file.

        Args:
            path_or_buf: String path or file-like object where the CSV will be saved.
            **kwargs: Additional arguments passed to pandas.DataFrame.to_csv().

        Returns:
            String path if a path was specified and return_path is True, otherwise None.

        """
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
            base_url=self._api_url or cfg.api_url,
            api_key=self._api_key or cfg.api_key,
        )
        await self._load_remote_evaluators(api)
        weights = await self._prepare_eval_weights()

        self.project = await self._get_or_create_project(api, self._project_name or cfg.project_name)
        self._project_name = None

        metadata = (self.metadata or {}).copy()
        metadata.update(weights)
        self.experiment = await self._create_experiment(
            api, self.project.id, self._experiment_name, self.tags, metadata
        )
        self._experiment_name = None

        ctx = build_context(
            service=self._service or cfg.service,
            project_name=self.project.name,
            app=None,
            experiment_id=self.experiment.id,
            experiment_name=self.experiment.name,
            api_url=self._api_url or cfg.api_url,
            otel_endpoint=self._otel_endpoint or cfg.otel_endpoint,
            otel_exporter_otlp_protocol=self._otel_exporter_otlp_protocol or cfg.otel_exporter_otlp_protocol,
            api_key=self._api_key or cfg.api_key,
            client_http=client_http,
            client_http_async=client_http_async,
            timeout_s=self._timeout_s or cfg.timeout_s,
            integrations=self._integrations,
        )

        with context._CTX_PAT.using(ctx):
            dataset = await self._prepare_dataset(self._raw_dataset)
        self._raw_dataset = None
        self.dataset = dataset

        self._prepared = True
        return ctx

    async def _load_remote_evaluators(self, api: PatronusAPIClient):
        for link_dict in self._chain:
            for evaluator_adapter in link_dict.get("evaluators", []):
                if isinstance(evaluator_adapter, StructuredEvaluatorAdapter):
                    evaluator = evaluator_adapter.evaluator
                    if isinstance(evaluator, AsyncRemoteEvaluator):
                        await evaluator.load(api=api)
                    elif isinstance(evaluator, RemoteEvaluator):
                        evaluator.load(api=api)

    async def _prepare_eval_weights(self):
        weights = {}
        for link_dict in self._chain:
            for evaluator in link_dict.get("evaluators", []):
                canonical_name = evaluator.canonical_name
                current_weight = evaluator.weight

                if current_weight is not None:
                    # Convert weight to string for consistent comparison and storage
                    current_weight_str = str(current_weight)

                    if canonical_name in weights:
                        # Compare the stored weight with current weight
                        stored_weight_str = str(weights[canonical_name])
                        if stored_weight_str != current_weight_str:
                            raise TypeError(
                                f"You cannot set different weights for the same evaluator: `{canonical_name}`. "
                                f"Found weights: {stored_weight_str} and {current_weight_str}"
                            )
                    else:
                        weights[canonical_name] = current_weight

        return {"evaluator_weights": weights}

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
        api: PatronusAPIClient, project_id: str, experiment_name: str, tags: Tags, metadata: Optional[dict[str, Any]]
    ) -> api_types.Experiment:
        name = generate_experiment_name(experiment_name)
        return await api.create_experiment(
            api_types.CreateExperimentRequest(project_id=project_id, name=name, tags=tags, metadata=metadata)
        )

    async def _run_chain(self, idx: int, row: datasets.Row):
        tracer = get_tracer()

        @contextlib.contextmanager
        def chain_link_span(create_span: bool, link_idx: int):
            if not create_span:
                yield
                return
            attrs = {Attributes.span_type.value: SpanTypes.experiment_chain_step.value}
            with tracer.start_as_current_span(f"experiment.chain.step.{link_idx}", attributes=attrs) as span:
                yield span

        attrs = {Attributes.span_type.value: SpanTypes.experiment_sample.value}
        with tracer.start_as_current_span("experiment.sample.processing", attributes=attrs):
            parent = None

            for link_idx, eval_link in enumerate(self._chain):
                with chain_link_span(create_span=len(self._chain) > 1, link_idx=link_idx):
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

                        if not isinstance(result, EvaluationResult):
                            raise TypeError(
                                f"evaluator {adapter} returned unexpected unexpected type {type(result)!r}. "
                                f"Allowed types: {EvaluationResult.__name__!r}."
                            )
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

            with bundled_eval("experiment.evaluation"):
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
    attributes = {Attributes.span_type: SpanTypes.experiment_task.value}
    return traced(f"experiment.task {task.__name__}", attributes=attributes)(task)
