# Patronus Python SDK

The Patronus Python SDK is a Python library for systematic evaluation of Large Language Models (LLMs).
Build, test, and improve your LLM applications with customizable tasks, evaluators, and comprehensive experiment tracking.

**Note:** This library is currently in **beta** and is not stable. The APIs may change in future releases.

## Documentation

For detailed documentation, including API references and advanced usage, please visit our [documentation](https://docs.patronus.ai/docs/experimentation-framework).

## Installation

```shell
pip install patronus
```

## Quickstart

### Tracing

```python
import patronus

patronus.init()

# Wrap function with @traced() decorator.
@patronus.traced()
def main():
    perform()

def perform():
    # Or use context start_span context manager.
    with patronus.start_span("Performing action"):
        # Do work
        ...

```

### Custom evaluations

```python
from patronus import init
from patronus import evaluator

init()

@evaluator()
def iexact_match(actual: str, expected: str) -> bool:
    return actual.lower().strip() == expected.lower().strip()

iexact_match("bonne nuit", "Bonne nuit")
```

### Patronus evaluations
```python
from patronus import init
from patronus import RemoteEvaluator

init()

check_hallucinates = RemoteEvaluator("lynx", "patronus:hallucination")

resp = check_hallucinates.evaluate(
    task_input="What is the car insurance policy?",
    task_context=(
        """
        To qualify for our car insurance policy, you need a way to show competence
        in driving which can be accomplished through a valid driver's license.
        You must have multiple years of experience and cannot be graduating from driving school before or on 2028.
        """
    ),
    task_output="To even qualify for our car insurance policy, you need to have a valid driver's license that expires later than 2028."
)
print(resp.model_dump_json(indent=4))
```

### Experiments

The Patronus Python SDK includes a powerful experimentation framework designed to help you evaluate, compare, and improve your AI models.
Whether you're working with pre-trained models, fine-tuning your own, or experimenting with new architectures,
this framework provides the tools you need to set up, execute, and analyze experiments efficiently.

```python
from patronus.evals import evaluator, RemoteEvaluator
from patronus.experiments import run_experiment, Row, TaskResult, FuncEvaluatorAdapter


def my_task(row: Row, **kwargs):
    return f"{row.task_input} World"


# Reference remote Judge Patronus Evaluator with is-concise criteria.
# This evaluator runs remotely on Patronus infrastructure.
is_concise = RemoteEvaluator("judge", "patronus:is-concise")


@evaluator()
def exact_match(row: Row, task_result: TaskResult, **kwargs):
    print(f"{task_result.output=}  :: {row.task_output=}")
    return task_result.output == row.task_output


result = run_experiment(
    project_name="Tutorial Project",
    dataset=[
        {
            "task_input": "Hello",
            "gold_answer": "Hello World",
        },
    ],
    task=my_task,
    evaluators=[is_concise, FuncEvaluatorAdapter(exact_match)],
)

result.to_csv("./experiment.csv")
```
