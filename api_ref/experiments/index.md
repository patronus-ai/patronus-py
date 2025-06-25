# experiments

## patronus.experiments

### adapters

#### BaseEvaluatorAdapter

Bases: `ABC`

Abstract base class for all evaluator adapters.

Evaluator adapters provide a standardized interface between the experiment framework and various types of evaluators (function-based, class-based, etc.).

All concrete adapter implementations must inherit from this class and implement the required abstract methods.

#### EvaluatorAdapter

```python
EvaluatorAdapter(evaluator: Evaluator)

```

Bases: `BaseEvaluatorAdapter`

Adapter for class-based evaluators conforming to the Evaluator or AsyncEvaluator protocol.

This adapter enables the use of evaluator classes that implement either the Evaluator or AsyncEvaluator interface within the experiment framework.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `evaluator` | `Union[Evaluator, AsyncEvaluator]` | The evaluator instance to adapt. |

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
    dataset=[{"task_output": "string        ", "gold_answer": "string"}],
    evaluators=[MatchAdapter(exact_match), MatchAdapter(fuzzy_match)],
)

```

Source code in `src/patronus/experiments/adapters.py`

```python
def __init__(self, evaluator: evals.Evaluator):
    if not isinstance(evaluator, evals.Evaluator):
        raise TypeError(f"{evaluator} is not {evals.Evaluator.__name__}.")
    self.evaluator = evaluator

```

##### transform

```python
transform(
    row: Row,
    task_result: Optional[TaskResult],
    parent: EvalParent,
    **kwargs: Any,
) -> tuple[list[typing.Any], dict[str, typing.Any]]

```

Transform experiment framework arguments to evaluation method arguments.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `row` | `Row` | The data row being evaluated. | *required* | | `task_result` | `Optional[TaskResult]` | The result of the task execution, if available. | *required* | | `parent` | `EvalParent` | The parent evaluation context. | *required* | | `**kwargs` | `Any` | Additional keyword arguments from the experiment. | `{}` |

Returns:

| Type | Description | | --- | --- | | `list[Any]` | A list of positional arguments to pass to the evaluator function. | | `dict[str, Any]` | A dictionary of keyword arguments to pass to the evaluator function. |

Source code in `src/patronus/experiments/adapters.py`

```python
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

```

##### evaluate

```python
evaluate(
    row: Row,
    task_result: Optional[TaskResult],
    parent: EvalParent,
    **kwargs: Any,
) -> EvaluationResult

```

Evaluate the given row and task result using the adapted evaluator function.

This method implements the BaseEvaluatorAdapter.evaluate() protocol.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `row` | `Row` | The data row being evaluated. | *required* | | `task_result` | `Optional[TaskResult]` | The result of the task execution, if available. | *required* | | `parent` | `EvalParent` | The parent evaluation context. | *required* | | `**kwargs` | `Any` | Additional keyword arguments from the experiment. | `{}` |

Returns:

| Type | Description | | --- | --- | | `EvaluationResult` | An EvaluationResult containing the evaluation outcome. |

Source code in `src/patronus/experiments/adapters.py`

```python
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

```

#### StructuredEvaluatorAdapter

```python
StructuredEvaluatorAdapter(
    evaluator: Union[
        StructuredEvaluator, AsyncStructuredEvaluator
    ],
)

```

Bases: `EvaluatorAdapter`

Adapter for structured evaluators.

Source code in `src/patronus/experiments/adapters.py`

```python
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

```

#### FuncEvaluatorAdapter

```python
FuncEvaluatorAdapter(
    fn: Callable[..., Any],
    weight: Optional[Union[str, float]] = None,
)

```

Bases: `BaseEvaluatorAdapter`

Adapter class that allows using function-based evaluators with the experiment framework.

This adapter serves as a bridge between function-based evaluators decorated with `@evaluator()` and the experiment framework's evaluation system. It handles both synchronous and asynchronous evaluator functions.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `fn` | `Callable` | The evaluator function to be adapted. |

Notes

- The function passed to this adapter must be decorated with `@evaluator()`.
- The adapter automatically handles the conversion between function results and proper evaluation result objects.

Examples:

````text
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
````

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

````

Source code in `src/patronus/experiments/adapters.py`

```python
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

````

##### transform

```python
transform(
    row: Row,
    task_result: Optional[TaskResult],
    parent: EvalParent,
    **kwargs: Any,
) -> tuple[list[typing.Any], dict[str, typing.Any]]

```

Transform experiment framework parameters to evaluator function parameters.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `row` | `Row` | The data row being evaluated. | *required* | | `task_result` | `Optional[TaskResult]` | The result of the task execution, if available. | *required* | | `parent` | `EvalParent` | The parent evaluation context. | *required* | | `**kwargs` | `Any` | Additional keyword arguments from the experiment. | `{}` |

Returns:

| Type | Description | | --- | --- | | `list[Any]` | A list of positional arguments to pass to the evaluator function. | | `dict[str, Any]` | A dictionary of keyword arguments to pass to the evaluator function. |

Source code in `src/patronus/experiments/adapters.py`

```python
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

```

##### evaluate

```python
evaluate(
    row: Row,
    task_result: Optional[TaskResult],
    parent: EvalParent,
    **kwargs: Any,
) -> EvaluationResult

```

Evaluate the given row and task result using the adapted evaluator function.

This method implements the BaseEvaluatorAdapter.evaluate() protocol.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `row` | `Row` | The data row being evaluated. | *required* | | `task_result` | `Optional[TaskResult]` | The result of the task execution, if available. | *required* | | `parent` | `EvalParent` | The parent evaluation context. | *required* | | `**kwargs` | `Any` | Additional keyword arguments from the experiment. | `{}` |

Returns:

| Type | Description | | --- | --- | | `EvaluationResult` | An EvaluationResult containing the evaluation outcome. |

Source code in `src/patronus/experiments/adapters.py`

```python
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

```

### experiment

#### Tags

```python
Tags = dict[str, str]

```

Tags are key-value pairs applied to experiments, task results and evaluation results.

#### Task

```python
Task = Union[
    TaskProtocol[Union[TaskResult, str, None]],
    TaskProtocol[Awaitable[Union[TaskResult, str, None]]],
]

```

A function that processes each dataset row and produces output for evaluation.

#### ExperimentDataset

```python
ExperimentDataset = Union[
    Dataset,
    DatasetLoader,
    list[dict[str, Any]],
    tuple[dict[str, Any], ...],
    DataFrame,
    Awaitable,
    Callable[[], Awaitable],
]

```

Any object that would "resolve" into Dataset.

#### TaskProtocol

Bases: `Protocol[T]`

Defines an interface for a task.

Task is a function that processes each dataset row and produces output for evaluation.

##### __call__

```python
__call__(*, row: Row, parent: EvalParent, tags: Tags) -> T

```

Processes a dataset row, using the provided context to produce task output.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `row` | `Row` | The dataset row to process. | *required* | | `parent` | `EvalParent` | Reference to the parent task's output and evaluation results. | *required* | | `tags` | `Tags` | Key-value pairs. | *required* |

Returns:

| Type | Description | | --- | --- | | `T` | Task output of type T or None to skip the row processing. |

Example

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

Source code in `src/patronus/experiments/experiment.py`

````python
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

````

#### ChainLink

Bases: `TypedDict`

Represents a single stage in an experiment's processing chain.

Each ChainLink contains an optional task function that processes dataset rows and a list of evaluators that assess the task's output.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `task` | `Optional[Task]` | Function that processes a dataset row and produces output. | | `evaluators` | `list[AdaptableEvaluators]` | List of evaluators to assess the task's output. |

#### Experiment

```python
Experiment(
    *,
    dataset: Any,
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
    integrations: Optional[list[Any]] = None,
    **kwargs,
)

```

Manages evaluation experiments across datasets using tasks and evaluators.

An experiment represents a complete evaluation pipeline that processes a dataset using defined tasks, applies evaluators to the outputs, and collects the results. Experiments track progress, create reports, and interface with the Patronus platform.

Create experiment instances using the create() class method or through the run_experiment() convenience function.

Source code in `src/patronus/experiments/experiment.py`

```python
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

```

##### create

```python
create(
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
    integrations: Optional[list[Any]] = None,
    **kwargs: Any,
) -> te.Self

```

Creates an instance of the class asynchronously with the specified parameters while performing necessary preparations. This method initializes various attributes including dataset, task, evaluators, chain, and additional configurations for managing concurrency, project details, service information, API keys, timeout settings, and integrations.

Use run_experiment for more convenient usage.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `dataset` | `ExperimentDataset` | The dataset to run evaluations against. | *required* | | `task` | `Optional[Task]` | A function that processes each dataset row and produces output for evaluation. Mutually exclusive with the chain parameter. | `None` | | `evaluators` | `Optional[list[AdaptableEvaluators]]` | A list of evaluators to assess the task output. Mutually exclusive with the chain parameter. | `None` | | `chain` | `Optional[list[ChainLink]]` | A list of processing stages, each containing a task and associated evaluators. Use this for multi-stage evaluation pipelines. | `None` | | `tags` | `Optional[Tags]` | Key-value pairs. All evaluations created by the experiment will contain these tags. | `None` | | `metadata` | `Optional[dict[str, Any]]` | Arbitrary dict. Metadata associated with the experiment. | `None` | | `max_concurrency` | `int` | Maximum number of concurrent task and evaluation operations. | `10` | | `project_name` | `Optional[str]` | Name of the project to create or use. Falls back to configuration or environment variables if not provided. | `None` | | `experiment_name` | `Optional[str]` | Custom name for this experiment run. A timestamp will be appended. | `None` | | `service` | `Optional[str]` | OpenTelemetry service name for tracing. Falls back to configuration or environment variables if not provided. | `None` | | `api_key` | `Optional[str]` | API key for Patronus services. Falls back to configuration or environment variables if not provided. | `None` | | `api_url` | `Optional[str]` | URL for the Patronus API. Falls back to configuration or environment variables if not provided. | `None` | | `otel_endpoint` | `Optional[str]` | OpenTelemetry collector endpoint. Falls back to configuration or environment variables if not provided. | `None` | | `otel_exporter_otlp_protocol` | `Optional[str]` | OpenTelemetry exporter protocol (grpc or http/protobuf). Falls back to configuration or environment variables if not provided. | `None` | | `ui_url` | `Optional[str]` | URL for the Patronus UI. Falls back to configuration or environment variables if not provided. | `None` | | `timeout_s` | `Optional[int]` | Timeout in seconds for API operations. Falls back to configuration or environment variables if not provided. | `None` | | `integrations` | `Optional[list[Any]]` | A list of OpenTelemetry instrumentors for additional tracing capabilities. | `None` | | `**kwargs` | `Any` | Additional keyword arguments passed to the experiment. | `{}` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `Experiment` | `Self` | ... |

Source code in `src/patronus/experiments/experiment.py`

```python
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

```

##### run

```python
run() -> te.Self

```

Executes the experiment by processing all dataset items.

Runs the experiment's task chain on each dataset row, applying evaluators to the results and collecting metrics. Progress is displayed with a progress bar and results are logged to the Patronus platform.

Returns:

| Type | Description | | --- | --- | | `Self` | The experiment instance. |

Source code in `src/patronus/experiments/experiment.py`

```python
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

```

##### to_dataframe

```python
to_dataframe() -> pd.DataFrame

```

Converts experiment results to a pandas DataFrame.

Creates a tabular representation of all evaluation results with dataset identifiers, task information, evaluation scores, and metadata.

Returns:

| Type | Description | | --- | --- | | `DataFrame` | A pandas DataFrame containing all experiment results. |

Source code in `src/patronus/experiments/experiment.py`

```python
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

```

##### to_csv

```python
to_csv(
    path_or_buf: Union[str, Path, IO[AnyStr]], **kwargs: Any
) -> Optional[str]

```

Saves experiment results to a CSV file.

Converts experiment results to a DataFrame and saves them as a CSV file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `path_or_buf` | `Union[str, Path, IO[AnyStr]]` | String path or file-like object where the CSV will be saved. | *required* | | `**kwargs` | `Any` | Additional arguments passed to pandas.DataFrame.to_csv(). | `{}` |

Returns:

| Type | Description | | --- | --- | | `Optional[str]` | String path if a path was specified and return_path is True, otherwise None. |

Source code in `src/patronus/experiments/experiment.py`

```python
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

```

#### run_experiment

```python
run_experiment(
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
    integrations: Optional[list[Any]] = None,
    **kwargs,
) -> Union[Experiment, typing.Awaitable[Experiment]]

```

Create and run an experiment.

This function creates an experiment with the specified configuration and runs it to completion. The execution handling is context-aware:

- When called from an asynchronous context (with a running event loop), it returns an awaitable that must be awaited.
- When called from a synchronous context (no running event loop), it blocks until the experiment completes and returns the Experiment object.

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

See Experiment.create for list of arguments.

Returns:

| Name | Type | Description | | --- | --- | --- | | `Experiment` | `Experiment` | In a synchronous context: the completed Experiment object. | | `Experiment` | `Awaitable[Experiment]` | In an asynchronous context: an awaitable that resolves to the Experiment object. |

Notes

For manual control of the event loop, you can create and run the experiment as follows:

```python
experiment = await Experiment.create(...)
await experiment.run()

```

Source code in `src/patronus/experiments/experiment.py`

````python
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

````

### types

#### EvalParent

```python
EvalParent = Optional[_EvalParent]

```

Type alias representing an optional reference to an evaluation parent, used to track the hierarchy of evaluations and their results

#### TaskResult

Bases: `BaseModel`

Represents the result of a task with optional output, metadata, and tags.

This class is used to encapsulate the result of a task, including optional fields for the output of the task, metadata related to the task, and any tags that can provide additional information or context about the task.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `output` | `Optional[str]` | The output of the task, if any. | | `metadata` | `Optional[dict[str, Any]]` | Additional information or metadata associated with the task. | | `tags` | `Optional[dict[str, str]]` | Key-value pairs used to tag and describe the task. |

#### EvalsMap

Bases: `dict`

A specialized dictionary for storing evaluation results with flexible key handling.

This class extends dict to provide automatic key normalization for evaluation results, allowing lookup by evaluator objects, strings, or any object with a canonical_name attribute.
