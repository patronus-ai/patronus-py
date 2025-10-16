# Patronus Evaluators

Patronus provides a suite of evaluators that help you assess LLM outputs without writing complex evaluation logic.
These managed evaluators run on Patronus infrastructure. Visit Patronus Platform console to define your own criteria.

## Using Patronus Evaluators

You can use Patronus evaluators through the `RemoteEvaluator` class:

```python
from patronus import init
from patronus.evals import RemoteEvaluator

init()

factual_accuracy = RemoteEvaluator("judge", "factual-accuracy")

# Evaluate an LLM output
result = factual_accuracy.evaluate(
    task_input="What is the capital of France?",
    task_output="The capital of France is Paris, which is located on the Seine River.",
    gold_answer="Paris"
)

print(f"Passed: {result.pass_}")
print(f"Score: {result.score}")
print(f"Explanation: {result.explanation}")
```

## Retry Configuration

RemoteEvaluators support automatic retry with exponential backoff for transient failures. You can configure the retry behavior using the following parameters:

```python
from patronus.evals import RemoteEvaluator

# Configure retry behavior
evaluator = RemoteEvaluator(
    "judge",
    "factual-accuracy",
    retry_max_attempts=5,       # Maximum number of retry attempts (default: 3)
    retry_initial_delay=2,      # Initial delay in seconds before first retry (default: 1)
    retry_backoff_factor=3      # Multiplier for delay between retries (default: 2)
)

result = evaluator.evaluate(
    task_input="What is the capital of France?",
    task_output="Paris",
    gold_answer="Paris"
)
```

**Retry Parameters:**

- `retry_max_attempts` (int): Maximum number of attempts to retry the evaluation request. Default is 3.
- `retry_initial_delay` (int): Initial delay in seconds before the first retry attempt. Default is 1 second.
- `retry_backoff_factor` (int): Multiplier applied to the delay between each retry attempt, creating exponential backoff. Default is 2.

**Example Retry Behavior:**

With default settings (`retry_max_attempts=3`, `retry_initial_delay=1`, `retry_backoff_factor=2`):
- 1st attempt: immediate
- 2nd attempt: after 1 second
- 3rd attempt: after 2 seconds (1 × 2)
- 4th attempt: after 4 seconds (2 × 2)

If all retry attempts fail, the original exception will be raised.

## Synchronous and Asynchronous Versions

Patronus evaluators are available in both synchronous and asynchronous versions:

```python
# Synchronous usage (as shown above)
factual_accuracy = RemoteEvaluator("judge", "factual-accuracy")
result = factual_accuracy.evaluate(...)

# Asynchronous usage
from patronus.evals import AsyncRemoteEvaluator

async_factual_accuracy = AsyncRemoteEvaluator("judge", "factual-accuracy")
result = await async_factual_accuracy.evaluate(...)
```
