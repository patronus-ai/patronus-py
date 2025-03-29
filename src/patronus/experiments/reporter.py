import statistics

import pandas as pd
import typing
import asyncio
import sys
import traceback
from typing import Optional

from .types import TaskResult
from .tqdm import AsyncTQDMWithHandle

from patronus import datasets
from patronus.exceptions import WithStackTraceMixin
from patronus.evals import EvaluationResult


class Result(typing.NamedTuple):
    link_idx: int
    task_name: Optional[str]
    task_result: Optional[TaskResult]
    evaluator_id: str
    criteria: Optional[str]
    evaluation_result: Optional[EvaluationResult]
    row: datasets.Row


class Reporter:
    tqdm: Optional[AsyncTQDMWithHandle]

    def __init__(self):
        self.tqdm = None
        self.lock = asyncio.Lock()

        self.results: list[Result] = []
        self.df: Optional[pd.DataFrame] = None

    def set_tqdm(self, tqdm: AsyncTQDMWithHandle):
        self.tqdm = tqdm

    def add_task_error(self, exc: Exception, row: datasets.Row):
        stack_trace = self._get_stack_trace(exc)
        self.print_error(
            "\n".join(
                [stack_trace, f"Task failed on sample {(row.dataset_id, row.sid)=} with the following error: {exc}"]
            )
        )

    def add_evaluator_error(self, exc: Exception, row: datasets.Row, evaluator_id: str, criteria: Optional[str]):
        stack_trace = self._get_stack_trace(exc)
        self.print_error(
            "\n".join(
                [
                    stack_trace,
                    f"Evaluator {(evaluator_id, criteria)=} failed on sample "
                    f"{(row.dataset_id, row.sid)=} with the following error: {exc}",
                ]
            )
        )

    @staticmethod
    def _get_stack_trace(exc: Exception) -> Optional[str]:
        stack_trace = None
        if isinstance(exc, WithStackTraceMixin):
            stack_trace = exc.stack_trace
        if stack_trace is None:
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            tb_text = "".join(tb_lines)
            stack_trace = tb_text
        return stack_trace

    def print_error(self, message: str):
        if self.tqdm:
            self.tqdm.clear()
        print(message, file=sys.stderr)
        if self.tqdm:
            self.tqdm.display()

    async def add_result(
        self,
        link_idx: int,
        task_name: Optional[str],
        task_result: Optional[TaskResult],
        evaluator_id: str,
        criteria: Optional[str],
        evaluation_result: Optional[EvaluationResult],
        row: datasets.Row,
    ):
        async with self.lock:
            self.results.append(
                Result(
                    link_idx=link_idx,
                    task_name=task_name,
                    task_result=task_result,
                    evaluator_id=evaluator_id,
                    criteria=criteria,
                    evaluation_result=evaluation_result,
                    row=row,
                )
            )

    def to_dataframe(self) -> pd.DataFrame:
        if self.df is not None:
            return self.df

        data = []

        for result in self.results:
            row_data = {
                "link_idx": result.link_idx,
                "task.name": result.task_name,
                "evaluator_id": result.evaluator_id,
                "criteria": result.criteria,
            }

            if result.task_result:
                row_data.update(
                    {
                        "task.output": result.task_result.output,
                        "task.metadata": result.task_result.metadata,
                        "task.tags": result.task_result.tags,
                    }
                )
            else:
                row_data.update(
                    {
                        "task.output": None,
                        "task.metadata": None,
                        "task.tags": None,
                    }
                )

            if result.evaluation_result:
                row_data.update(
                    {
                        "eval.score": result.evaluation_result.score,
                        "eval.pass": result.evaluation_result.pass_,
                        "eval.text_output": result.evaluation_result.text_output,
                        "eval.metadata": result.evaluation_result.metadata,
                        "eval.explanation": result.evaluation_result.explanation,
                        "eval.tags": result.evaluation_result.tags,
                        "eval.evaluation_duration": result.evaluation_result.evaluation_duration,
                        "eval.explanation_duration": result.evaluation_result.explanation_duration,
                    }
                )
            else:
                row_data.update(
                    {
                        "eval.score": None,
                        "eval.pass": None,
                        "eval.text_output": None,
                        "eval.metadata": None,
                        "eval.explanation": None,
                        "eval.tags": None,
                        "eval.evaluation_duration": None,
                        "eval.explanation_duration": None,
                    }
                )

            if result.row:
                row_series = result.row.row

                for column_name, value in row_series.items():
                    # Avoid overwriting existing columns
                    if column_name not in row_data:
                        row_data[column_name] = value

            data.append(row_data)

        self.df = pd.DataFrame(data)
        # reclaim memory
        self.results = []
        return self.df

    def summary(self):
        # summary can be called only once experiment finished.
        # We set tqdm to None to prevent showing progress bar in between error messages
        self.tqdm = None
        df = self.to_dataframe()

        if df.empty:
            self.print_error("\nEvaluation summary could not be generated: No successful evaluations found.")
            self.print_error(
                "All evaluations in the experiment may have failed. See the error messages above for details."
            )
            return

        df["criteria"] = df["criteria"].fillna("None")

        grouped = df.groupby(["link_idx", "evaluator_id", "criteria"])

        def group_name(evaluator_id, criteria, link_idx):
            if criteria != "None":
                ev = f"{criteria} ({evaluator_id})"
            else:
                ev = evaluator_id
            return f"{ev} [link_idx={int(link_idx)}]"

        for (link_idx, evaluator_id, criteria), group in grouped:
            name = group_name(evaluator_id, criteria, link_idx)

            scores_series = group["eval.score"]
            passes_series = group["eval.pass"]

            scores_and_passes = list(zip(scores_series, passes_series))

            scores = [s for s, _ in scores_and_passes if s is not None]
            passes = [int(p) for _, p in scores_and_passes if p is not None]

            if scores:
                print_summary(name, scores, passes, len(scores_and_passes), display_hist=True)


def print_summary(name: str, scores: list[float], passes: list[int], count: int, display_hist: bool):
    print()
    print(name)
    print("-" * len(name))
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
