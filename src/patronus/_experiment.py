import logging
import asyncio
import datetime
import itertools
import os
import re
import statistics
import time
import typing
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor

from ._async_utils import run_until_complete
from . import _api as api
from ._client import Client
from ._evaluators import Evaluator, EvaluatorOutput
from ._tasks import Task, TaskResult

log = logging.getLogger(__name__)


class DatasetDatum(typing.TypedDict):
    evaluated_model_system_prompt: typing.NotRequired[str | None]
    evaluated_model_retrieved_context: typing.NotRequired[list[str] | None]
    evaluated_model_input: typing.NotRequired[str | None]
    evaluated_model_output: typing.NotRequired[str | None]
    evaluated_model_gold_answer: typing.NotRequired[str | None]


class DefaultReporter:
    # batch_size 10 is max batch size that Patronus AI API accepts in single call
    batch_size = 10

    def __init__(self, client: Client, project_name: str, flush_interval: int = 10):
        self._client = client
        self.project_name = project_name

        self._lock = asyncio.Lock()
        self.last_export_slice = slice(0, 0)
        self.last_flush_ts = time.time()
        self.flush_interval = flush_interval

        self.evaluators = set()
        self.remote_results: list[api.ExportEvaluationResult] = []
        self.outgoing_results: list[api.ExportEvaluationResult] = []

    async def add(
        self,
        evaluator_name: str,
        profile_name: str | None,
        input_datum: dict,
        task_result: TaskResult,
        evaluation_result: EvaluatorOutput,
        tags: dict[str, str],
        *,
        already_captured: bool,
    ):
        entry = api.ExportEvaluationResult(
            app=self.project_name,
            evaluator_id=evaluator_name,
            profile_name=profile_name,
            evaluated_model_system_prompt=task_result.evaluated_model_system_prompt
            or input_datum.get("evaluated_model_system_prompt"),
            evaluated_model_retrieved_context=input_datum.get("evaluated_model_retrieved_context"),
            evaluated_model_input=input_datum.get("evaluated_model_input"),
            evaluated_model_output=task_result.evaluated_model_output,
            evaluated_model_gold_answer=input_datum.get("evaluated_model_gold_answer"),
            pass_=evaluation_result.result.pass_,
            score_raw=evaluation_result.result.score_raw,
            evaluation_duration=datetime.timedelta(seconds=evaluation_result.duration),
            evaluated_model_name=task_result.evaluated_model_name,
            evaluated_model_provider=task_result.evaluated_model_provider,
            evaluated_model_params=task_result.evaluated_model_params,
            evaluated_model_selected_model=task_result.evaluated_model_selected_model,
            tags=tags,
        )
        async with self._lock:
            self.evaluators.add((evaluator_name, profile_name))
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
            await self._client.api.export_evaluations(
                api.ExportEvaluationRequest(
                    evaluation_results=results_to_export,
                )
            )
            log.debug(f"Flushed {len(results_to_export)} results")
        self.last_flush_ts = time.time()

    def summary(self):
        for evaluator_name, profile_name in self.evaluators:
            name = evaluator_name if not profile_name else f"{evaluator_name}:{profile_name}"
            results: typing.Iterator[api.ExportEvaluationResult] = itertools.chain(
                self.outgoing_results, self.remote_results
            )
            results = filter(lambda r: r.evaluator_id == evaluator_name and r.profile_name == profile_name, results)
            scores_and_passes = list(map(lambda r: (r.score_raw, r.pass_), results))

            scores = [x[0] for x in scores_and_passes if x[0] is not None]
            passes = [int(x[1]) for x in scores_and_passes if x[1] is not None]

            print_summary(name, scores, passes, len(scores_and_passes), display_hist=True)


async def with_semaphore(sem, task):
    async with sem:
        return await task


class Experiment:
    project_name: str
    project_id: uuid.UUID | None
    experiment_id: uuid.UUID
    # TODO handle with API?
    experiment_name: str

    task: Task
    evaluators: list[Evaluator]

    _client: Client | None

    _pool: ThreadPoolExecutor
    _sem: asyncio.Semaphore

    def __init__(
        self,
        client: Client | None,
        project_name: str | None,
        data: list[DatasetDatum],
        task: Task,
        evaluators: list[Evaluator],
        tags: dict[str, str],
        max_concurrency: int,
        name: str = "",
    ):
        self._client = client
        self.project_name = project_name or "default"
        self.experiment_id = uuid.uuid4()
        self.experiment_name = generate_experiment_name(name)

        self.data = data
        self.task = task
        self.evaluators = evaluators
        self.tags = tags

        self._sem = asyncio.Semaphore(max_concurrency)
        self._pool = ThreadPoolExecutor()

        self.reporter = DefaultReporter(client, self.project_name, flush_interval=10)

    async def prepare(self):
        # TODO prepare() should initialize experiment doing pre-required API calls
        #      such as checking API caller, creating project & experiment etc.
        ...
        # if self._client:
        #     project = self._client.get_project_by_name(self.project_name)
        #     self.project_id = project.id

    async def run(self):
        title = f"Running experiment: {self.project_name}/{self.experiment_name}"
        print("=" * len(title))
        print(title)
        print("=" * len(title))

        start_date = datetime.datetime.utcnow().isoformat() + "Z"

        tasks = []
        for datum in self.data:
            task = self.run_task_and_eval(datum)
            tasks.append(asyncio.create_task(with_semaphore(self._sem, task)))

        for t in tasks:
            await t

        await self.reporter.flush()

        end_date = (datetime.datetime.utcnow() + datetime.timedelta(hours=1)).isoformat() + "Z"

        self.reporter.summary()

        print()
        print(get_link(self.project_name, start_date, end_date))

    async def run_task_and_eval(self, datum):
        loop = asyncio.get_running_loop()

        em_system_prompt = datum.get("evaluated_model_system_prompt")
        em_retrieved_context = datum.get("evaluated_model_retrieved_context")
        em_input = datum.get("evaluated_model_input")
        em_gold_answer = datum.get("evaluated_model_gold_answer")

        task = await self.task.execute(
            loop,
            self._pool,
            evaluated_model_system_prompt=em_system_prompt,
            evaluated_model_retrieved_context=em_retrieved_context,
            evaluated_model_input=em_input,
            tags=self.tags,
        )

        outgoing_tags = self.tags
        if task.tags:
            outgoing_tags = {**self.tags, **task.tags}

        futures = [
            loop.create_task(
                evaluator.execute(
                    loop,
                    self._pool,
                    app=self.project_name,
                    evaluated_model_system_prompt=em_system_prompt,
                    evaluated_model_retrieved_context=em_retrieved_context,
                    evaluated_model_input=em_input,
                    evaluated_model_output=task.evaluated_model_output,
                    evaluated_model_gold_answer=em_gold_answer,
                    tags=outgoing_tags,
                )
            )
            for evaluator in self.evaluators
        ]

        for evaluator, f in zip(self.evaluators, futures):
            eval_result = await f

            await self.reporter.add(
                evaluator.name,
                evaluator.profile_name,
                datum,
                task,
                eval_result,
                outgoing_tags,
                already_captured=evaluator.remote_capture,
            )


def experiment(
    client: Client | None,
    name: str | None,
    data: list[DatasetDatum],
    task: Task,
    evaluators: list[Evaluator],
    tags: dict[str, str] | None = None,
    display_hist: bool = False,
):
    ex = Experiment(
        client=client,
        project_name=name,
        data=data,
        task=task,
        evaluators=evaluators,
        tags=tags or {},
        max_concurrency=10,
        name="",
    )

    async def run():
        await ex.prepare()
        await ex.run()

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
    print("Value Range".ljust(20), "Count".ljust(10), "Histogram")
    for i in range(bin_count):
        bin_start = min_val + i * bin_size
        bin_end = bin_start + bin_size
        bin_count = bins[i]
        bar = "#" * int(bin_count * scale_factor)
        print(f"{bin_start:.2f} - {bin_end:.2f}".ljust(20), f"{bin_count}".ljust(10), bar)


def get_link(app: str, start, end) -> str:
    params = {"projectId": app, "startDate": start, "endDate": end}
    return f"https://app.patronus.ai/monitoring?{urllib.parse.urlencode(params)}"


def generate_experiment_name(name: str) -> str:
    ts = int(time.time())
    if name:
        return f"{name}-{ts}"
    try:
        login = os.getlogin()
        return f"{login}-{ts}"
    except OSError:  # Possible in-cluster error: No such device or address
        return str(ts)
