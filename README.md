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

### Evaluation

For quick testing and exploration, you can use the synchronous evaluate() method:

```python
import os
from patronus import Client

client = Client(
    # This is the default and can be omitted
    api_key=os.environ.get("PATRONUS_API_KEY"),
)
result = client.evaluate(
    evaluator="lynx",
    criteria="patronus:hallucination",
    evaluated_model_input="Who are you?",
    evaluated_model_output="My name is Barry.",
    evaluated_model_retrieved_context="My name is John.",
)
print(f"Pass: {result.pass_}")
print(f"Explanation: {result.explanation}")
```

The Patronus Python SDK is designed to work primarily with async/await patterns,
which is the recommended way to use the library. Here's a feature-rich example using async evaluation:

```python
import asyncio
from patronus import Client

client = Client()

no_apologies = client.remote_evaluator(
    "judge",
    "patronus:no-apologies",
    explain_strategy="always",
    max_attempts=3,
)


async def evaluate():
    result = await no_apologies.evaluate(
        evaluated_model_input="How to kill a docker container?",
        evaluated_model_output="""
        I cannot assist with that question as it has been marked as inappropriate.
        I must respectfully decline to provide an answer."
        """,
    )
    print(f"Pass: {result.pass_}")
    print(f"Explanation: {result.explanation}")


asyncio.run(evaluate())
```

### Experiment

The Patronus Python SDK includes a powerful experimentation framework designed to help you evaluate, compare, and improve your AI models.
Whether you're working with pre-trained models, fine-tuning your own, or experimenting with new architectures,
this framework provides the tools you need to set up, execute, and analyze experiments efficiently.

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
