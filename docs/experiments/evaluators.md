# Using Evaluators in Experiments

Evaluators are the core assessment tools in Patronus experiments, measuring the quality of task outputs against defined criteria. This page covers how to use various types of evaluators in the Patronus Experimentation Framework.

## Evaluator Types

The framework supports several types of evaluators:

- **Remote Evaluators**: Use Patronus's managed evaluation services
- **Custom Evaluators**: Your own evaluation logic.
    - **Function-based**: Simple functions decorated with @evaluator() that need to be wrapped with FuncEvaluatorAdapter when used in experiments.
    - **Class-based**: More powerful evaluators created by extending `StructuredEvaluator` (synchronous) or `AsyncStructuredEvaluator` (asynchronous) base classes with predefined interfaces.

Each type has different capabilities and use cases.

## Remote Evaluators

Remote evaluators run on Patronus infrastructure and provide standardized, high-quality assessments:

```python
from patronus.evals import RemoteEvaluator
from patronus.experiments import run_experiment

experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[
        RemoteEvaluator("judge", "patronus:is-concise"),
        RemoteEvaluator("lynx", "patronus:hallucination"),
        RemoteEvaluator("judge", "patronus:is-helpful")
    ]
)
```

## Class-Based Evaluators

You can create custom evaluator classes by inheriting from the Patronus base classes:

> **Note**: The following example uses the `transformers` library from Hugging Face. Install it with `pip install transformers` before running this code.

```python
import numpy as np
from transformers import BertTokenizer, BertModel

from patronus import StructuredEvaluator, EvaluationResult
from patronus.experiments import run_experiment


class BERTScore(StructuredEvaluator):
    def __init__(self, pass_threshold: float):
        self.pass_threshold = pass_threshold
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def evaluate(self, *, task_output: str, gold_answer: str, **kwargs) -> EvaluationResult:
        output_toks = self.tokenizer(task_output, return_tensors="pt", padding=True, truncation=True)
        gold_answer_toks = self.tokenizer(gold_answer, return_tensors="pt", padding=True, truncation=True)

        output_embeds = self.model(**output_toks).last_hidden_state.mean(dim=1).detach().numpy()
        gold_answer_embeds = self.model(**gold_answer_toks).last_hidden_state.mean(dim=1).detach().numpy()

        score = np.dot(output_embeds, gold_answer_embeds.T) / (
            np.linalg.norm(output_embeds) * np.linalg.norm(gold_answer_embeds)
        )

        return EvaluationResult(
            score=score,
            pass_=score >= self.pass_threshold,
            tags={"pass_threshold": str(self.pass_threshold)},
        )


experiment = run_experiment(
    dataset=[
        {
            "task_output": "Translate 'Goodbye' to Spanish.",
            "gold_answer": "Adiós",
        }
    ],
    evaluators=[BERTScore(pass_threshold=0.8)],
)
```

Class-based evaluators that inherit from `StructuredEvaluator` or `AsyncStructuredEvaluator` are automatically adapted for use in experiments.

## Function Evaluators

For simpler evaluation logic, you can use function-based evaluators.
When using function evaluators in experiments, you must wrap them with `FuncEvaluatorAdapter`.

### Standard Function Adapter

By default, `FuncEvaluatorAdapter` expects functions that follow this interface:

```python
from typing import Optional
from patronus import evaluator
from patronus.datasets import Row
from patronus.experiments.types import TaskResult, EvalParent
from patronus.evals import EvaluationResult
from patronus.experiments import run_experiment, FuncEvaluatorAdapter

@evaluator()
def standard_evaluator(
    row: Row,
    task_result: TaskResult,
    parent: EvalParent,
    **kwargs
) -> Optional[EvaluationResult]:
    """
    Standard interface for function evaluators used with FuncEvaluatorAdapter.
    """
    if not task_result or not task_result.output:
        # Skip the evaluation
        return None

    if row.gold_answer and row.gold_answer.lower() in task_result.output.lower():
        return EvaluationResult(score=1.0, pass_=True, text_output="Contains answer")
    else:
        return EvaluationResult(score=0.0, pass_=False, text_output="Missing answer")

# Use with standard adapter
experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[
        FuncEvaluatorAdapter(standard_evaluator)
    ]
)
```

### Custom Function Adapters

If your evaluator function doesn't match the standard interface, you can create a custom adapter:

```python
from patronus import evaluator
from patronus.datasets import Row
from patronus.experiments.types import TaskResult, EvalParent
from patronus.experiments.adapters import FuncEvaluatorAdapter

# An evaluator function with a different interface
@evaluator()
def exact_match(expected: str, actual: str, case_sensitive: bool = False) -> bool:
    """
    Checks if actual text exactly matches expected text.
    """
    if not case_sensitive:
        return expected.lower() == actual.lower()
    return expected == actual

# Custom adapter to transform experiment arguments to evaluator arguments
class ExactMatchAdapter(FuncEvaluatorAdapter):
    def __init__(self, case_sensitive=False):
        super().__init__(exact_match)
        self.case_sensitive = case_sensitive

    def transform(
        self,
        row: Row,
        task_result: TaskResult,
        parent: EvalParent,
        **kwargs
    ) -> tuple[list, dict]:
        # Create arguments list and dict for the evaluator function
        args = []  # No positional arguments in this case

        # Create keyword arguments matching the evaluator's parameters
        evaluator_kwargs = {
            "expected": row.gold_answer,
            "actual": task_result.output if task_result else "",
            "case_sensitive": self.case_sensitive
        }

        return args, evaluator_kwargs

# Use custom adapter in an experiment
experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[
        ExactMatchAdapter(case_sensitive=False)
    ]
)
```

The `transform()` method is the key to adapting any function to the experiment framework.
It takes the standard arguments provided by the framework and transforms them into the format your evaluator function expects.

## Combining Evaluator Types

You can use multiple types of evaluators in a single experiment:

```python
experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[
        # Remote evaluator
        RemoteEvaluator("judge", "factual-accuracy"),

        # Class-based evaluator
        BERTScore(pass_threshold=0.7),

        # Function evaluator with standard adapter
        FuncEvaluatorAdapter(standard_evaluator),

        # Function evaluator with custom adapter
        ExactMatchAdapter(case_sensitive=False)
    ]
)
```

## Evaluator Chains

In multi-stage evaluation chains, evaluators from one stage can see the results of previous stages:

```python
experiment = run_experiment(
    dataset=dataset,
    chain=[
        # First stage
        {
            "task": generate_summary,
            "evaluators": [
                RemoteEvaluator("judge", "conciseness"),
                RemoteEvaluator("judge", "coherence")
            ]
        },
        # Second stage - evaluating based on first stage results
        {
            "task": None,  # No additional processing
            "evaluators": [
                # This evaluator can see previous evaluations
                DependentEvaluator()
            ]
        }
    ]
)

# Example of a function evaluator that uses previous results
@evaluator()
def final_aggregate_evaluator(row, task_result, parent, **kwargs):
    # Check if we have previous evaluation results
    if not parent or not parent.evals:
        return None

    # Access evaluations from previous stage
    conciseness = parent.evals.get("judge:conciseness")
    coherence = parent.evals.get("judge:coherence")

    # Use the previous results
    avg_score = ((conciseness.score or 0) + (coherence.score or 0)) / 2
    return EvaluationResult(score=avg_score, pass_=avg_score > 0.7)
```
## Best Practices

When using evaluators in experiments:

1. **Use the right evaluator type for the job**: Remote evaluators for standardized assessments, custom evaluators for specialized logic
2. **Focus each evaluator on one aspect**: Create multiple focused evaluators rather than one complex evaluator
3. **Provide detailed explanations**: Include explanations to help understand evaluation results
4. **Create custom adapters when needed**: Don't force your evaluator functions to match the standard interface if there's a more natural way to express them
5. **Handle edge cases gracefully**: Consider what happens with empty inputs, very long texts, etc.
6. **Reuse evaluators across experiments**: Create a library of evaluators for consistent assessment

Next, we'll explore advanced features of the Patronus Experimentation Framework.
