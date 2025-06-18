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
        RemoteEvaluator("judge", "factual-accuracy", weight=0.4),

        # Class-based evaluator
        BERTScore(pass_threshold=0.7, weight=0.3),

        # Function evaluator with standard adapter
        FuncEvaluatorAdapter(standard_evaluator, weight=0.2),

        # Function evaluator with custom adapter
        ExactMatchAdapter(case_sensitive=False, weight=0.1)
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
## Evaluator Weights (Experiments Only)

!!! note "Experiments Feature"
    Evaluator weights are only supported when using evaluators within the experiment framework. This feature is not available for standalone evaluator usage.

You can assign weights to evaluators to indicate their relative importance in your evaluation strategy. Weights can be provided as either strings or floats representing valid decimal numbers and are automatically stored as experiment metadata.

Weights work consistently across all evaluator types but are configured differently depending on whether you're using remote evaluators, function-based evaluators, or class-based evaluators.

### Weight Support by Evaluator Type

Each evaluator type handles weight configuration differently:

#### Remote Evaluators

For remote evaluators, pass the `weight` parameter directly to the `RemoteEvaluator` constructor:

```python
from patronus.evals import RemoteEvaluator
from patronus.experiments import run_experiment

# Remote evaluator with weight (string or float)
pii_evaluator = RemoteEvaluator("pii", "patronus:pii:1", weight="0.6")
conciseness_evaluator = RemoteEvaluator("judge", "patronus:is-concise", weight=0.4)

experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[pii_evaluator, conciseness_evaluator]
)
```

#### Function-Based Evaluators

For function-based evaluators, pass the `weight` parameter to the `FuncEvaluatorAdapter` that wraps your evaluator function:

```python
from patronus import evaluator
from patronus.experiments import FuncEvaluatorAdapter, run_experiment
from patronus.datasets import Row

@evaluator()
def exact_match(row: Row, **kwargs) -> bool:
    return row.task_output.lower().strip() == row.gold_answer.lower().strip()

# Function evaluator with weight (string or float)
exact_match_weighted = FuncEvaluatorAdapter(exact_match, weight=0.7)

experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[exact_match_weighted]
)
```

#### Class-Based Evaluators

For class-based evaluators, pass the `weight` parameter to your evaluator's constructor and ensure it's passed to the parent class:

```python
from typing import Union
from patronus import StructuredEvaluator, EvaluationResult
from patronus.experiments import run_experiment

class CustomEvaluator(StructuredEvaluator):
    def __init__(self, threshold: float, weight: Union[str, float] = None):
        super().__init__(weight=weight)  # Pass to parent class
        self.threshold = threshold

    def evaluate(self, *, task_output: str, **kwargs) -> EvaluationResult:
        score = len(task_output) / 100  # Simple length-based scoring
        return EvaluationResult(
            score=score,
            pass_=score >= self.threshold
        )

# Class-based evaluator with weight (string or float)
custom_evaluator = CustomEvaluator(threshold=0.5, weight=0.3)

experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[custom_evaluator]
)
```

### Complete Example

Here's a comprehensive example demonstrating weighted evaluators of all three types, based on the patterns shown in the experiment framework:

```python
from patronus.experiments import FuncEvaluatorAdapter, run_experiment
from patronus import RemoteEvaluator, EvaluationResult, StructuredEvaluator, evaluator
from patronus.datasets import Row

class DummyEvaluator(StructuredEvaluator):
    def evaluate(self, task_output: str, gold_answer: str, **kwargs) -> EvaluationResult:
        return EvaluationResult(score_raw=1, pass_=True)

@evaluator
def exact_match(row: Row, **kwargs) -> bool:
    return row.task_output.lower().strip() == row.gold_answer.lower().strip()

experiment = run_experiment(
    project_name="Weighted Evaluation Example",
    dataset=[
        {
            "task_input": "Please provide your contact details.",
            "task_output": "My email is john.doe@example.com and my phone number is 123-456-7890.",
            "gold_answer": "My email is john.doe@example.com and my phone number is 123-456-7890.",
        },
        {
            "task_input": "Share your personal information.",
            "task_output": "My name is Jane Doe and I live at 123 Elm Street.",
            "gold_answer": "My name is Jane Doe and I live at 123 Elm Street.",
        },
    ],
    evaluators=[
        RemoteEvaluator("pii", "patronus:pii:1", weight="0.3"),           # Remote evaluator with string weight
        FuncEvaluatorAdapter(exact_match, weight="0.3"),                   # Function evaluator with string weight
        DummyEvaluator(weight="0.4"),                                      # Class evaluator with string weight
    ],
    experiment_name="Weighted Evaluators Demo"
)
```

### Weight Validation and Rules

1. **Experiments Only**: Weights are exclusively available within the experiment framework - they cannot be used with standalone evaluator calls
2. **Valid Format**: Weights must be valid decimal numbers provided as either strings or floats (e.g., "0.3", 1.0, 0.7)
3. **Consistency**: The same evaluator (identified by its canonical name) cannot have different weights within the same experiment
4. **Automatic Storage**: Weights are automatically collected and stored in the experiment's metadata under the "evaluator_weights" key
5. **Optional**: Weights are optional - evaluators without weights will simply not have weight metadata stored
6. **Best Practice**: Consider making weights sum to 1.0 for clearer interpretation of relative importance

### Error Examples

```python
# Invalid weight format - will raise TypeError
RemoteEvaluator("judge", "patronus:is-concise", weight="invalid")
RemoteEvaluator("judge", "patronus:is-concise", weight=[1, 2, 3])  # Lists not supported

# Inconsistent weights for same evaluator - will raise TypeError during experiment
run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[
        RemoteEvaluator("judge", "patronus:is-concise", weight=0.7),
        RemoteEvaluator("judge", "patronus:is-concise", weight="0.3"),  # Different weight!
    ]
)
```

## Best Practices

When using evaluators in experiments:

1. **Use the right evaluator type for the job**: Remote evaluators for standardized assessments, custom evaluators for specialized logic
2. **Focus each evaluator on one aspect**: Create multiple focused evaluators rather than one complex evaluator
3. **Provide detailed explanations**: Include explanations to help understand evaluation results
4. **Create custom adapters when needed**: Don't force your evaluator functions to match the standard interface if there's a more natural way to express them
5. **Handle edge cases gracefully**: Consider what happens with empty inputs, very long texts, etc.
6. **Reuse evaluators across experiments**: Create a library of evaluators for consistent assessment
7. **Weight consistency across evaluator types**: When using evaluator weights, maintain consistency across experiments regardless of whether you're using remote, function-based, or class-based evaluators
8. **Consider weight distribution**: When using weights, consider making them sum to 1.0 for clearer interpretation of relative importance (e.g., "0.4", "0.3", "0.3" rather than "0.1", "0.1", "0.1")
9. **Document weight rationale**: Consider documenting why specific weights were chosen for your evaluation strategy, especially when mixing different evaluator types

Next, we'll explore advanced features of the Patronus Experimentation Framework.
