# Patronus Python SDK

[![PyPI version](https://img.shields.io/pypi/v/openai.svg)](https://pypi.org/project/patronus/)
[![Documentation](https://img.shields.io/badge/docs-link-blue.svg)](https://patronus-ai.github.io/patronus-py/)

---

**SDK Documentation**: <a href="https://patronus-ai.github.io/patronus-py" target="_blank">https://patronus-ai.github.io/patronus-py</a>

**Platform Documentation**: <a href="https://docs.patronus.ai" target="_blank">https://docs.patronus.ai</a>

---

The Patronus Python SDK is a Python library for systematic evaluation of Large Language Models (LLMs).
Build, test, and improve your LLM applications with customizable tasks, evaluators, and comprehensive experiment tracking.

## Documentation

For detailed documentation, including API references and advanced usage, please visit our [documentation](https://docs.patronus.ai/docs/experimentation-framework).

## Installation

```shell
pip install patronus
```

## Quickstart

### Initialization

```python
import patronus

# Initialize with your Patronus API key
patronus.init(
    project_name="My Agent",  # Optional, defaults to "Global"
    api_key="your-api-key"      # Optional, can also be set via environment variable
)
```

You can also use a configuration file (patronus.yaml) for initialization:

```yaml
# patronus.yaml
api_key: "your-api-key"
project_name: "My Agent"
```

With this configuration file in your working directory, you can simply call:

```python
import patronus
patronus.init()  # Automatically loads config from patronus.yaml
```

### Tracing

```python
import patronus

patronus.init()

# Trace a function with the @traced decorator
@patronus.traced()
def process_input(user_query):
    # Process the input
    return "Processed: " + user_query

# Use context manager for finer-grained tracing
def complex_operation():
    with patronus.start_span("Data preparation"):
        # Prepare data
        pass

    with patronus.start_span("Model inference"):
        # Run model
        pass
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
print(f"""
Hallucination evaluation:
Passed: {resp.pass_}
Score: {resp.score}
Explanation: {resp.explanation}
""")
```

### User-Defined Evaluators

```python
from patronus import init, evaluator
from patronus.evals import EvaluationResult

init()

# Simple evaluator function
@evaluator()
def exact_match(actual: str, expected: str) -> bool:
    return actual.strip() == expected.strip()

# More complex evaluator with detailed result
@evaluator()
def semantic_match(actual: str, expected: str) -> EvaluationResult:
    similarity = calculate_similarity(actual, expected)  # Your similarity function
    return EvaluationResult(
        score=similarity,
        pass_=similarity > 0.8,
        text_output="High similarity" if similarity > 0.8 else "Low similarity",
        explanation=f"Calculated similarity: {similarity}"
    )

# Use the evaluators
result = exact_match("Hello world", "Hello world")
print(f"Match: {result}")  # Output: Match: True
```

### Running Experiments

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
