# Patronus LLM Evaluation library


Patronus is a Python library developed by [Patronus AI](https://www.patronus.ai/)
that provides a robust framework and utility functions for evaluating Large Language Models (LLMs).
This library simplifies the process of running and scoring evaluations across different LLMs,
making it easier for developers to benchmark model performance on various tasks.

**Note:** This library is currently in **beta** and is not stable. The APIs may change in future releases.

**Note:** This library requires Python 3.11 or greater.

## Features

- **Modular Evaluation Framework:** Easily plug in different models and evaluation/scoring mechanisms.
- **Seamless Integration with Patronus AI Platform:** Effortlessly connect with the Patronus AI platform to run evaluations and export results.
- **Custom Evaluators:** Use built-in evaluators, create your own based on various scoring methods, or leverage our state-of-the-art remote evaluators.

## Documentation

For detailed documentation, including API references and advanced usage, please visit our [documentation](https://docs.patronus.ai/docs/experimentation-framework).

## Installation

```shell
pip install patronus
```

## Quickstart

```python
import os
from patronus import Client, task, evaluator

client = Client(
    # This is the default and can be omitted
    api_key=os.environ.get("PATRONUSAI_API_KEY"),
)

@task
def hello_world_task(evaluated_model_input: str) -> str:
    return f"{evaluated_model_input} World"

@evaluator
def exact_match(evaluated_model_output: str, evaluated_model_gold_answer: str) -> bool:
    return evaluated_model_output == evaluated_model_gold_answer

client.experiment(
    "Tutorial Project",
    data=[
        {
            "evaluated_model_input": "Hello",
            "evaluated_model_gold_answer": "Hello World",
        },
    ],
    task=hello_world_task,
    evaluators=[exact_match],
)
```
