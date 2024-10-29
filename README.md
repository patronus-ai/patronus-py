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

### Experiment

```python
import os
from patronus import Client, Row, TaskResult, evaluator, task

client = Client(
    # This is the default and can be omitted
    api_key=os.environ.get("PATRONUS_API_KEY"),
)


@task
def my_task(row: Row):
    return f"{row.evaluated_model_input} World"


@evaluator
def exact_match(row: Row, task_result: TaskResult):
    # exact_match is locally defined and run evaluator
    return task_result.evaluated_model_output == row.evaluated_model_gold_answer


# Reference remote Judge Patronus Evaluator with is-concise criteria.
# This evaluator runs remotely on Patronus infrastructure.
is_concise = client.remote_evaluator("judge", "patronus:is-concise")

client.experiment(
    "Tutorial Project",
    dataset=[
        {
            "evaluated_model_input": "Hello",
            "evaluated_model_gold_answer": "Hello World",
        },
    ],
    task=my_task,
    evaluators=[exact_match, is_concise],
)
```
