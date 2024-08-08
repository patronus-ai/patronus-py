import datetime
import re
import statistics
import urllib.parse

from tqdm import tqdm

from ._evaluators import Evaluator, EvaluatorOutput
from ._tasks import Task
from ._client import Client
from . import _api as api


def experiment(
    client: Client | None,
    name: str | None,
    # TODO type data properly
    data: list[dict],
    task: Task,
    evaluators: list[Evaluator],
    tags: dict[str, str] | None = None,
    display_hist: bool = False,
):
    name = gen_name(name)
    tags = tags or {}

    title = f"Running experiment: {name}"
    print("=" * len(title))
    print(title)
    print("=" * len(title))

    start_date = datetime.datetime.utcnow().isoformat() + "Z"

    for evaluator in evaluators:
        evaluator: Evaluator

        outputs: list[EvaluatorOutput] = []

        print(f"\nEvaluator: {evaluator.name}")

        for datum in tqdm(data):
            em_system_prompt = datum.get("evaluated_model_system_prompt")
            em_retrieved_context = datum.get("evaluated_model_retrieved_context")
            em_input = datum.get("evaluated_model_input")
            em_gold_answer = datum.get("evaluated_model_gold_answer")

            task_result = task(
                evaluated_model_system_prompt=em_system_prompt,
                evaluated_model_retrieved_context=em_retrieved_context,
                evaluated_model_input=em_input,
                tags=tags,
            )

            outgoing_tags = tags
            if task_result.tags:
                outgoing_tags = {**tags, **task_result.tags}

            eval_output = evaluator(
                app=name,
                evaluated_model_system_prompt=em_system_prompt,
                evaluated_model_retrieved_context=em_retrieved_context,
                evaluated_model_input=em_input,
                evaluated_model_output=task_result.evaluated_model_output,
                evaluated_model_gold_answer=em_gold_answer,
                tags=outgoing_tags,
            )
            outputs.append(eval_output)

            if eval_output.result.tags:
                outgoing_tags = {**outgoing_tags, **eval_output.result.tags}

            if evaluator.remote_capture:
                continue

            if not client:
                continue

            client.api.export_evaluations(
                api.ExportEvaluationRequest(
                    evaluation_results=[
                        api.ExportEvaluationResult(
                            app=name,
                            evaluator_id=evaluator.name,
                            evaluated_model_system_prompt=task_result.evaluated_model_system_prompt or em_system_prompt,
                            evaluated_model_retrieved_context=em_retrieved_context,
                            evaluated_model_input=em_input,
                            evaluated_model_output=task_result.evaluated_model_output,
                            evaluated_model_gold_answer=em_gold_answer,
                            pass_=eval_output.result.pass_,
                            score_raw=eval_output.result.score_raw,
                            evaluation_duration=datetime.timedelta(seconds=eval_output.duration),
                            evaluated_model_name=task_result.evaluated_model_name,
                            evaluated_model_provider=task_result.evaluated_model_provider,
                            evaluated_model_params=task_result.evaluated_model_params,
                            evaluated_model_selected_model=task_result.evaluated_model_selected_model,
                            tags=outgoing_tags,
                        )
                    ]
                )
            )

        print_summary(evaluator.name, outputs, display_hist)

    end_date = (datetime.datetime.utcnow() + datetime.timedelta(hours=1)).isoformat() + "Z"

    print()
    print(get_link(name, start_date, end_date))


def print_summary(evaluator_name: str, outputs: list[EvaluatorOutput], display_hist: bool):
    rs: list[EvaluatorOutput]
    scores = [x.result.score_raw for x in outputs if x.result.score_raw is not None]
    passes = [int(x.result.pass_) for x in outputs if x.result.pass_ is not None]

    title = f"{evaluator_name} ({len(outputs)} samples)"

    print()
    print(title)
    print("-" * len(title))
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


def print_histogram(data, bin_count=10):
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
    scale_factor = 50 / max_bin_count  # Scale the histogram to a max width of 50 characters

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
