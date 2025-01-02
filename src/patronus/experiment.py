import asyncio
import datetime
import inspect
import itertools
import logging
import os
import re
import statistics
import sys
import time
import traceback
import typing
import urllib.parse
import warnings
from concurrent.futures import ThreadPoolExecutor
from sys import version_info
from typing import Optional, Union

import pandas as pd
import typing_extensions as te
from tqdm.asyncio import tqdm as tqdm_async

from . import api_types, types
from .async_utils import run_until_complete
from .client import Client
from .config import config
from .datasets import Dataset, DatasetLoader, Row
from .evaluators import Evaluator
from .evaluators_remote import RemoteEvaluator
from .retry import retry
from .tasks import Task, nop_task

log = logging.getLogger(__name__)


class EvalLink(typing.TypedDict, total=False):
    task: Optional[Task]
    evaluators: list[Evaluator]


class AsyncTQDMWithHandle(tqdm_async):
    """
    Workaround for accessing tqdm instance with async tasks.
    Instead of calling gather which don't provide access to tqdm instance:
    ```
    tqdm_async.gather(features)
    ```

    Call prep_gather() follow by gather()
    ```
    tqdm_instance = AsyncTQDMWithHandle.pre_gather(features)
    ...
    tqdm_instance.gather()
    ```

    tqdm_instance can be used to clear and display progress bar using tqdm_instance.clear() and tqdm_instance.display()
    methods.
    """

    async def gather(self):
        def into_iter():
            yield from self

        res = [await f for f in into_iter()]
        return [i for _, i in sorted(res)]

    @classmethod
    async def prep_gather(cls, *fs, loop=None, timeout=None, total=None, **tqdm_kwargs) -> te.Self:
        async def wrap_awaitable(i, f):
            return i, await f

        ifs = [wrap_awaitable(i, f) for i, f in enumerate(fs)]
        return cls.prep_as_completed(ifs, loop=loop, timeout=timeout, total=total, **tqdm_kwargs)

    @classmethod
    def prep_as_completed(cls, fs, *, loop=None, timeout=None, total=None, **tqdm_kwargs):
        if total is None:
            total = len(fs)
        kwargs = {}
        if version_info[:2] < (3, 10):
            kwargs["loop"] = loop
        return cls(
            asyncio.as_completed(fs, timeout=timeout, **kwargs),
            total=total,
            **tqdm_kwargs,
        )


class ReportedTaskResult(typing.NamedTuple):
    sid: int
    link_idx: int
    task_name: str
    task_result: types.TaskResult


class ReportedEvaluationResult(typing.NamedTuple):
    sid: int
    link_idx: int
    evaluator_id: str
    criteria: Optional[str]
    evaluation_result: api_types.ExportEvaluationResult
    evaluation_metadata: Optional[dict[str, typing.Any]]
    evaluation_explanation: Optional[str]


class Reporter:
    # batch_size 10 is max batch size that Patronus AI API accepts in single call
    batch_size = 10
    tqdm = None

    def __init__(self, client: Client, experiment_id: str, flush_interval: int = 10):
        self._client = client
        self.experiment_id = experiment_id

        self._lock = asyncio.Lock()
        self.last_export_slice = slice(0, 0)
        self.last_flush_ts = time.time()
        self.flush_interval = flush_interval

        self.task_results: list[ReportedTaskResult] = []

        EvaluatorIDT = str
        CriteriaT = str
        self.evaluators: set[tuple[EvaluatorIDT, CriteriaT]] = set()
        self.remote_results: list[ReportedEvaluationResult] = []
        self.outgoing_results: list[ReportedEvaluationResult] = []

        self.task_errors = []
        self.evaluator_errors = []

    def set_tqdm(self, tqdm: AsyncTQDMWithHandle):
        self.tqdm = tqdm

    def add_task_result(self, sid: int, link_idx: int, task_name: str, r: types.TaskResult):
        self.task_results.append(ReportedTaskResult(sid=sid, link_idx=link_idx, task_name=task_name, task_result=r))

    async def add_evaluation_result(
        self,
        evaluator_name: str,
        criteria: Optional[str],
        link_idx: int,
        row: Row,
        task_result: Optional[types.TaskResult],
        evaluation_result: types.EvaluatorOutput,
        tags: dict[str, str],
        *,
        dataset_id: str,
        dataset_sample_id: int,
        already_captured: bool,
    ):
        model_output = (task_result and task_result.evaluated_model_output) or row.evaluated_model_output
        task_metadata = (task_result and task_result.metadata) or {}

        eval_metadata = {}
        evaluation_duration = None
        explanation_duration = None
        if isinstance(evaluation_result.result, types.EvaluationResult):
            eval_metadata = evaluation_result.result.metadata
            edur = evaluation_result.result.evaluation_duration_s
            if not edur:
                edur = evaluation_result.duration
            evaluation_duration = datetime.timedelta(seconds=edur)
            if evaluation_result.result.explanation_duration_s:
                explanation_duration = datetime.timedelta(seconds=evaluation_result.result.explanation_duration_s)
        elif isinstance(evaluation_result.result, api_types.EvaluationResult):
            eval_metadata = evaluation_result.result.evaluation_metadata
            evaluation_duration = evaluation_result.result.evaluation_duration
            explanation_duration = evaluation_result.result.explanation_duration

        entry = ReportedEvaluationResult(
            sid=dataset_sample_id,
            link_idx=link_idx,
            evaluator_id=evaluator_name,
            criteria=criteria,
            evaluation_result=api_types.ExportEvaluationResult(
                experiment_id=self.experiment_id,
                evaluator_id=evaluator_name,
                criteria=criteria,
                evaluated_model_system_prompt=row.evaluated_model_system_prompt,
                evaluated_model_retrieved_context=row.evaluated_model_retrieved_context,
                evaluated_model_input=row.evaluated_model_input,
                evaluated_model_output=model_output,
                evaluated_model_gold_answer=row.evaluated_model_gold_answer,
                evaluated_model_attachments=row.evaluated_model_attachments,
                pass_=evaluation_result.result.pass_,
                score_raw=evaluation_result.result.score_raw,
                text_output=evaluation_result.result.text_output,
                explanation=evaluation_result.result.explanation,
                evaluation_duration=evaluation_duration,
                explanation_duration=explanation_duration,
                evaluation_metadata=eval_metadata,
                evaluated_model_name=task_metadata.get("evaluated_model_name"),
                evaluated_model_provider=task_metadata.get("evaluated_model_provider"),
                evaluated_model_params=task_metadata.get("evaluated_model_params"),
                evaluated_model_selected_model=task_metadata.get("evaluated_model_selected_model"),
                dataset_id=dataset_id,
                dataset_sample_id=dataset_sample_id,
                tags=tags,
            ),
            evaluation_metadata=eval_metadata,
            evaluation_explanation=evaluation_result.result.explanation,
        )

        async with self._lock:
            self.evaluators.add((evaluator_name, criteria))
            if already_captured:
                self.remote_results.append(entry)
            else:
                self.outgoing_results.append(entry)

        await self._conditional_flush()

    async def _conditional_flush(self):
        async with self._lock:
            buffered = len(self.outgoing_results) - self.last_export_slice.stop
            if buffered == 0:
                return
            if buffered >= self.batch_size or self.last_flush_ts + self.flush_interval < time.time():
                await self._flush()

    async def flush(self):
        async with self._lock:
            buffered = len(self.outgoing_results) - self.last_export_slice.stop
            if buffered == 0:
                return
            await self._flush()

    async def _flush(self):
        while self.last_export_slice.stop < len(self.outgoing_results):
            upper_idx = min(
                len(self.outgoing_results),
                self.last_export_slice.stop + self.batch_size,
            )
            self.last_export_slice = slice(self.last_export_slice.stop, upper_idx)
            results_to_export = self.outgoing_results[self.last_export_slice]

            @retry(max_attempts=5, initial_delay=2)
            async def call():
                await self._client.api.export_evaluations(
                    api_types.ExportEvaluationRequest(
                        evaluation_results=list(e.evaluation_result for e in results_to_export),
                    )
                )
                log.debug(f"Exported {len(results_to_export)} results")

            await call()
        self.last_flush_ts = time.time()

    async def task_error(self, err: Exception, datum, dataset_data_id):
        stack_trace = getattr(err, "stack_trace", None)
        if stack_trace is None:
            stack_trace = traceback.format_exc()
        self.print_error(stack_trace)
        self.print_error(f"Task failed on sample {dataset_data_id!r} with the  following error: {err}")
        self.task_errors.append((err, datum, dataset_data_id))

    async def evaluator_error(self, err: Exception, datum, dataset_data_id, evaluator: str, criteria: str):
        stack_trace = getattr(err, "stack_trace", None)
        if stack_trace is None:
            stack_trace = traceback.format_exc()
        self.print_error(stack_trace)
        self.print_error(
            f"Evaluator ({evaluator}, {criteria}) failed on sample {dataset_data_id} with the following error: {err}"
        )
        self.evaluator_errors.append((err, datum, dataset_data_id, evaluator, criteria))

    def print_error(self, message: str):
        if self.tqdm:
            self.tqdm.clear()
        print(message, file=sys.stderr)
        if self.tqdm:
            self.tqdm.display()

    def summary(self):
        for evaluator_name, criteria in self.evaluators:
            name = evaluator_name if not criteria else f"{evaluator_name}:{criteria}"
            results: typing.Iterator[ReportedEvaluationResult] = itertools.chain(
                self.outgoing_results, self.remote_results
            )
            results = filter(
                lambda r: r.evaluation_result.evaluator_id == evaluator_name
                and r.evaluation_result.criteria == criteria,
                results,
            )
            scores_and_passes = list(
                map(
                    lambda r: (
                        r.evaluation_result.score_raw,
                        r.evaluation_result.pass_,
                    ),
                    results,
                )
            )

            scores = [x[0] for x in scores_and_passes if x[0] is not None]
            passes = [int(x[1]) for x in scores_and_passes if x[1] is not None]

            print_summary(name, scores, passes, len(scores_and_passes), display_hist=True)

        if self.task_errors or self.evaluator_errors:
            print()
        if self.task_errors:
            print(f"Task failures: {len(self.task_errors)}")
        if self.evaluator_errors:
            print(f"Evaluators failures: {len(self.evaluator_errors)}")

    def task_results_df(self) -> pd.DataFrame:
        if not self.task_results:
            columns = [
                "sid",
                "link_idx",
                "task_name",
                "task_result_evaluated_model_output",
                "task_result_metadata",
            ]
            return pd.DataFrame(columns=columns).set_index(["sid", "link_idx"])

        def dict_with_sid(r: ReportedTaskResult) -> dict:
            return {
                "sid": r.sid,
                "link_idx": r.link_idx,
                "task_name": r.task_name,
                "task_result_evaluated_model_output": r.task_result.evaluated_model_output,
                "task_result_metadata": r.task_result.metadata,
            }

        df = pd.DataFrame.from_records(map(dict_with_sid, self.task_results)).sort_values(
            ["sid", "link_idx", "task_name"]
        )
        df.set_index(["sid", "link_idx"], inplace=True)
        return df

    def evaluation_results_df(self):
        if not self.remote_results and not self.outgoing_results:
            return pd.DataFrame()
        all_eval_results = itertools.chain(self.remote_results, self.outgoing_results)

        def mapper(r: ReportedEvaluationResult) -> dict:
            return {
                "sid": r.sid,
                "link_idx": r.link_idx,
                "evaluator_id": r.evaluator_id,
                "criteria": r.criteria,
                "pass": r.evaluation_result.pass_,
                "score_raw": r.evaluation_result.score_raw,
                "evaluation_duration": r.evaluation_result.evaluation_duration,
                "evaluation_metadata": r.evaluation_metadata,
                "evaluation_explanation": r.evaluation_explanation,
                "tags": r.evaluation_result.tags,
            }

        df = pd.DataFrame.from_records(map(mapper, all_eval_results))
        if df.empty:
            return df
        df = df.sort_values(["sid", "link_idx", "evaluator_id", "criteria"])
        df.set_index("sid", inplace=True)
        return df


async def with_semaphore(sem, coro):
    async with sem:
        return await coro


class Experiment:
    project_name: str
    project_id: Optional[str]
    experiment_id: Optional[str]
    # TODO handle with API?
    experiment_name: str
    whoami: Optional[api_types.WhoAmIResponse]

    __raw_dataset: typing.Any
    dataset: Optional[Dataset]
    chain: list[EvalLink]

    tags: dict[str, str]

    reporter: Optional[Reporter]

    _client: Optional[Client]

    _pool: ThreadPoolExecutor
    _sem: asyncio.Semaphore

    def __init__(
        self,
        client: Optional[Client],
        project_name: Optional[str],
        dataset,  # TODO type hint
        chain: list[EvalLink],
        tags: Optional[dict[str, str]],
        max_concurrency: int,
        experiment_name: str = "",
        experiment_id: Optional[str] = None,
        **kwargs,
    ):
        self._client = client
        self.project_id = None
        self.project_name = re.sub(r"[^a-zA-Z0-9\-_ ]", "-", project_name.strip()) or "default"
        self.experiment_id = experiment_id
        self.experiment_name = generate_experiment_name(experiment_name)

        self.__raw_dataset = dataset
        self.dataset = None
        self.chain = chain

        self.tags = tags or {}

        self._sem = asyncio.Semaphore(max_concurrency)
        self._pool = ThreadPoolExecutor()

        self.reporter = None

    async def prepare(self):
        for eval_link in self.chain:
            if (task := eval_link.get("task")) and not isinstance(task, Task):
                raise ValueError(f"task {task!r} must inherit from Task. Did you forget to use @task decorator?")

        print("Preparing dataset... ", end="")
        self.dataset = await self.fetch_dataset(self.__raw_dataset)
        print("DONE")

        print("Preparing evaluators... ", end="")
        for eval_link in self.chain:
            evaluators = eval_link.get("evaluators", [])
            for e in evaluators:
                if not isinstance(e, Evaluator):
                    raise ValueError(
                        f"evaluator {e!r} must inherit from Evaluator. Did you forget to use @evaluator decorator?"
                    )

            # Preload all remote evaluators
            preloads = [e.load() for e in evaluators if isinstance(e, RemoteEvaluator)]
            if preloads:
                await asyncio.gather(*preloads)

        print("DONE")

        if not self._client:
            return

        self.whoami = await self._client.api.whoami()

        if not self.experiment_id:
            project = await self._client.api.create_project(api_types.CreateProjectRequest(name=self.project_name))
            self.project_id = project.id
            self.project_name = project.name

            ex = await self._client.api.create_experiment(
                api_types.CreateExperimentRequest(project_id=self.project_id, name=self.experiment_name, tags=self.tags)
            )
            self.experiment_id = ex.id
        else:
            ex = await self._client.api.get_experiment(self.experiment_id)
            if ex is None:
                raise ValueError(f"experiment with id {self.experiment_id!r} not found")
            project = await self._client.api.get_project(ex.project_id)
            self.experiment_name = ex.name
            self.project_id = project.id
            self.project_name = project.name

        # TODO associate reported that doesn't need client if client not available
        self.reporter = Reporter(self._client, self.experiment_id, flush_interval=10)

    @classmethod
    async def fetch_dataset(cls, dataset) -> Dataset:
        if isinstance(dataset, Dataset):
            return dataset
        elif isinstance(dataset, DatasetLoader):
            return await dataset.load()
        elif isinstance(dataset, (list, tuple)):
            return Dataset.from_records(dataset)
        elif isinstance(dataset, pd.DataFrame):
            return Dataset.from_dataframe(dataset)
        elif inspect.iscoroutine(dataset):
            return await cls.fetch_dataset(await dataset)
        elif inspect.iscoroutinefunction(dataset):
            return await cls.fetch_dataset(await dataset())
        elif callable(dataset):
            return await cls.fetch_dataset(dataset())
        else:
            raise ValueError(f"'dataset' passed to the experiment is an unexpected object of type {type(dataset)!r}")

    async def run(self):
        title = f"Experiment  {self.project_name}/{self.experiment_name}"
        print("=" * len(title))

        tasks = []
        for row in self.dataset.iterrows():
            task = self.run_task_and_eval(row, dataset_id=self.dataset.dataset_id, dataset_sample_id=row.sid)
            tasks.append(task)

        tqdm = await AsyncTQDMWithHandle.prep_gather(*tasks, desc=title, unit="sample")
        self.reporter.set_tqdm(tqdm)
        await tqdm.gather()

        await self.reporter.flush()

        self.reporter.summary()

        print()
        print(get_link(self.whoami.caller.api_key.account.id, self.experiment_id))

    async def run_task_and_eval(self, row: Row, dataset_id: Optional[str], dataset_sample_id: int):
        loop = asyncio.get_running_loop()

        parent = types._EvalParent(task=None, evals=None, parent=None)

        for link_idx, eval_link in enumerate(self.chain):
            task = eval_link.get("task")
            evaluators = eval_link.get("evaluators", [])

            outgoing_tags = self.tags
            task_result = None

            if task:
                try:
                    task_result = await task.execute(
                        loop,
                        self._pool,
                        row=row,
                        tags=self.tags,
                        parent=parent,
                    )
                except Exception as e:
                    await self.reporter.task_error(e, row, dataset_sample_id)
                    return

                if task_result is None:
                    parent = types._EvalParent(task=None, evals=None, parent=parent)
                    # If task returned non it means the record processing should be skipped
                    return

                self.reporter.add_task_result(dataset_sample_id, link_idx, task.name, task_result)

                if task_result.tags:
                    outgoing_tags = merge_tags({}, task_result.tags, experiment_tags=self.tags)

            futures = [
                loop.create_task(
                    with_semaphore(
                        self._sem,
                        evaluator.execute(
                            loop=loop,
                            executor=self._pool,
                            row=row,
                            task_result=task_result,
                            tags=outgoing_tags,
                            parent=parent,
                            experiment_id=self.experiment_id,
                            dataset_id=dataset_id,
                            dataset_sample_id=dataset_sample_id,
                        ),
                    )
                )
                for evaluator in evaluators
            ]

            has_eval_errors = False
            eval_results_map = types.EvalsMap()

            for evaluator, f in zip(evaluators, futures):
                try:
                    eval_result: Optional[types.EvaluatorOutput] = await f
                except Exception as e:
                    await self.reporter.evaluator_error(
                        e,
                        row,
                        dataset_sample_id,
                        evaluator.name,
                        evaluator.criteria,
                    )
                    eval_results_map[evaluator.display_name()] = None
                    has_eval_errors = True
                    continue

                eval_results_map[evaluator.display_name()] = eval_result.result if eval_result else None
                if eval_result is None:
                    # If evaluator returned None it means the evaluation has been skipped
                    continue

                eval_tags = eval_result.result.tags or {}
                outgoing_tags = merge_tags(outgoing_tags, eval_tags, experiment_tags=self.tags)

                await self.reporter.add_evaluation_result(
                    evaluator.name,
                    evaluator.criteria,
                    link_idx,
                    row,
                    task_result,
                    eval_result,
                    outgoing_tags,
                    dataset_id=dataset_id,
                    dataset_sample_id=dataset_sample_id,
                    already_captured=evaluator.remote_capture,
                )

            if has_eval_errors:
                return
            parent = types._EvalParent(task=task_result, evals=eval_results_map, parent=parent)

    def to_dataframe(self) -> pd.DataFrame:
        df_dataset = self.dataset.df.set_index("sid", inplace=False)
        df_task_results = self.reporter.task_results_df()
        df_evaluation_results = self.reporter.evaluation_results_df()
        df = df_evaluation_results.join(df_task_results, on=["sid", "link_idx"], how="left")
        df = df.join(df_dataset, on="sid", how="left")
        return df

    def to_dataset(
        self,
        *,
        dataset_id: Optional[str] = None,
        rename_columns: Optional[dict[str, str]] = None,
    ) -> Dataset:
        df = self.to_dataframe()
        if rename_columns:
            col_to_drop = [col for col in rename_columns.keys() if col in df.columns]
            df.drop(columns=col_to_drop, inplace=True)
            df.rename(columns=rename_columns, inplace=True)
        return Dataset.from_dataframe(df, dataset_id=dataset_id)

    def to_csv(self, path_or_buf, sep: str = ",", **kwargs) -> Optional[str]:
        df = self.to_dataframe()
        return df.to_csv(path_or_buf, sep, **kwargs)


def experiment(
    client: Optional[Client],
    project_name: Optional[str],
    *,
    dataset,
    task: Optional[Task] = None,
    evaluators: Optional[list[Evaluator]] = None,
    #
    chain: Optional[list[EvalLink]] = None,
    tags: Optional[dict[str, str]] = None,
    experiment_name: str = "",
    max_concurrency: int = 10,
    **kwargs,
) -> Union[typing.Awaitable[Experiment], Experiment]:
    if chain:
        assert (
            task is nop_task and not evaluators
        ), "Passing 'chain' argument prevents use of 'task' and 'evaluators' arguments."
    if dataset is None:
        dataset = kwargs.get("data")

    if not chain:
        chain = [{"task": task, "evaluators": evaluators}]

    ex = Experiment(
        client=client,
        project_name=project_name,
        dataset=dataset,
        chain=chain,
        tags=tags or {},
        max_concurrency=max_concurrency,
        experiment_name=experiment_name,
        **kwargs,
    )

    async def run() -> Experiment:
        await ex.prepare()
        await ex.run()
        return ex

    return run_until_complete(run())


def print_summary(name: str, scores: list[float], passes: list[int], count: int, display_hist: bool):
    title = f"Summary: {name}"

    print()
    print(title)
    print("-" * len(title))
    print(f"Count     : {count}")
    print(f"Pass rate : {round(statistics.mean(passes), 3)}")
    print(f"Mean      : {round(statistics.mean(scores), 3)}")
    print(f"Min       : {round(min(scores), 3)}")
    print(f"25%       : {round(percentile(scores, 25), 3)}")
    print(f"50%       : {round(percentile(scores, 50), 3)}")
    print(f"75%       : {round(percentile(scores, 75), 3)}")
    print(f"Max       : {round(max(scores), 3)}")

    if display_hist:
        print()
        print("Score distribution")
        print_histogram(scores)


def gen_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]", "", name).lower()
    ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    name = name or "unknown"
    return f"ex-{name}-{ts}"


def percentile(data: list[float], p: int):
    data = sorted(data)
    index = (p / 100) * (len(data) - 1)
    if index.is_integer():
        return data[int(index)]
    else:
        lower_bound = int(index)
        upper_bound = lower_bound + 1
        weight = index - lower_bound
        return data[lower_bound] * (1 - weight) + data[upper_bound] * weight


def print_histogram(data, bin_count=5):
    # Calculate the range of the data
    min_val = min(data)
    max_val = max(data)

    if min_val == max_val:
        if min_val > 0.5:
            min_val = 0
        else:
            max_val = 1

    range_val = max_val - min_val

    # Calculate bin size
    bin_size = range_val / bin_count

    # Initialize bins
    bins = [0] * bin_count

    # Distribute data into bins
    for value in data:
        # Find the appropriate bin for the current value
        bin_index = int((value - min_val) / bin_size)
        # Edge case for the maximum value
        if bin_index == bin_count:
            bin_index -= 1
        bins[bin_index] += 1

    # Determine the width of the histogram
    max_bin_count = max(bins)
    scale_factor = 20 / max_bin_count  # Scale the histogram to a max width of 50 characters

    # Print the histogram
    print("Score Range".ljust(20), "Count".ljust(10), "Histogram")
    for i in range(bin_count):
        bin_start = min_val + i * bin_size
        bin_end = bin_start + bin_size
        bin_count = bins[i]
        bar = "#" * int(bin_count * scale_factor)
        print(f"{bin_start:.2f} - {bin_end:.2f}".ljust(20), f"{bin_count}".ljust(10), bar)


def get_link(account_id: str, experiment_id: str) -> str:
    params = {"account_id": account_id}
    ui_url = config().ui_url.rstrip("/")
    return f"{ui_url}/experiments/{experiment_id}?{urllib.parse.urlencode(params)}"


def generate_experiment_name(name: str) -> str:
    ts = int(time.time())
    if name:
        return f"{name}-{ts}"
    try:
        login = os.getlogin()
        return f"{login}-{ts}"
    except OSError:  # Possible in-cluster error: No such device or address
        return str(ts)


def merge_tags(tags: dict, new_tags: dict, experiment_tags: dict) -> dict:
    tags = {**tags, **new_tags}
    common_keys = set(tags.keys()) & set(experiment_tags.keys())
    diff = {key for key in common_keys if tags[key] != experiment_tags[key]}
    if diff:
        warnings.warn(f"Overriding experiment tags is not allowed. Tried to override tag: {list(diff)!r}")
    return {**tags, **experiment_tags}
