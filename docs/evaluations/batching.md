# Batch Evaluations

When evaluating multiple outputs or using multiple evaluators, Patronus provides efficient batch evaluation capabilities.
This page covers how to perform batch evaluations and manage evaluation groups.

## Using Patronus Client

For more advanced batch evaluation needs, use the `Patronus` client:

```python
from patronus import init
from patronus.pat_client import Patronus
from patronus.evals import RemoteEvaluator

init()

with Patronus() as client:
    # Run multiple evaluators in parallel
    results = client.evaluate(
        evaluators=[
            RemoteEvaluator("judge", "patronus:is-helpful"),
            RemoteEvaluator("lynx", "patronus:hallucination")
        ],
        task_input="What is quantum computing?",
        task_output="Quantum computing uses quantum bits or qubits to perform computations...",
        gold_answer="Computing that uses quantum phenomena like superposition and entanglement"
    )

    # Check if all evaluations passed
    if results.all_succeeded():
        print("All evaluations passed!")
    else:
        print("Some evaluations failed:")
        for failed in results.failed_evaluations():
            print(f"  - {failed.text_output}")
```

The `Patronus` client provides:

- Parallel evaluation execution
- Connection pooling
- Error handling
- Result aggregation

### Asynchronous Evaluation

For asynchronous workflows, use `AsyncPatronus`:

```python
import asyncio
from patronus import init
from patronus.pat_client import AsyncPatronus
from patronus.evals import AsyncRemoteEvaluator

init()


async def evaluate_responses():
    async with AsyncPatronus() as client:
        # Run evaluations asynchronously
        results = await client.evaluate(
            evaluators=[
                AsyncRemoteEvaluator("judge", "patronus:is-helpful"),
                AsyncRemoteEvaluator("lynx", "patronus:hallucination")
            ],
            task_input="What is quantum computing?",
            task_output="Quantum computing uses quantum bits or qubits to perform computations...",
            gold_answer="Computing that uses quantum phenomena like superposition and entanglement"
        )

        print(f"Number of evaluations: {len(results.results)}")
        print(f"All passed: {results.all_succeeded()}")

# Run the async function
asyncio.run(evaluate_responses())
```

## Background Evaluation

For non-blocking evaluation, use the `evaluate_bg()` method:

```python
from patronus import init
from patronus.pat_client import Patronus
from patronus.evals import RemoteEvaluator

init()

with Patronus() as client:
    # Start background evaluation
    future = client.evaluate_bg(
        evaluators=[
            RemoteEvaluator("judge", "factual-accuracy"),
            RemoteEvaluator("judge", "patronus:helpfulness")
        ],
        task_input="Explain how vaccines work.",
        task_output="Vaccines work by training the immune system to recognize and combat pathogens..."
    )

    # Do other work while evaluation happens in background
    print("Continuing with other tasks...")

    results = future.get()  # Blocks until complete

    print(f"Evaluation complete: {results.all_succeeded()}")
```

The async version works similarly:

```python
async with AsyncPatronus() as client:
    # Start background evaluation
    task = client.evaluate_bg(
        evaluators=[...],
        task_input="...",
        task_output="..."
    )

    # Do other async work
    await some_other_async_function()

    # Get results when needed
    results = await task
```

## Working with Evaluation Results

The `evaluate()` method returns an `EvaluationContainer` with several useful methods:

```python
results = client.evaluate(evaluators=[...], task_input="...", task_output="...")

if results.any_failed():
    print("Some evaluations failed")

if results.all_succeeded():
    print("All evaluations passed")

for failed in results.failed_evaluations():
    print(f"Failed: {failed.text_output}")

for success in results.succeeded_evaluations():
    print(f"Passed: {success.text_output}")

if results.has_exception():
    results.raise_on_exception()  # Re-raise any exceptions that occurred
```

## Example: Comprehensive Quality Check

Here's a complete example of batch evaluation for content quality:

```python
from patronus import init
from patronus.pat_client import Patronus
from patronus.evals import RemoteEvaluator

init()

def check_content_quality(question, answer):
    with Patronus() as client:
        results = client.evaluate(
            evaluators=[
                RemoteEvaluator("judge", "factual-accuracy"),
                RemoteEvaluator("judge", "helpfulness"),
                RemoteEvaluator("judge", "coherence"),
                RemoteEvaluator("judge", "grammar"),
                RemoteEvaluator("lynx", "patronus:hallucination")
            ],
            task_input=question,
            task_output=answer
        )

        if results.any_failed():
            print("Content quality check failed")
            for failed in results.failed_evaluations():
                print(f"- Failed check: {failed.text_output}")
                print(f"  Explanation: {failed.explanation}")
            return False

        print("Content passed all quality checks")
        return True

check_content_quality(
    "What is the capital of France?",
    "The capital of France is Paris, which is located on the Seine River."
)
```

## Using the `bundled_eval()` Context Manager

The `bundled_eval()` is a lower-level context manager that groups multiple evaluations together based on their arguments.
This is particularly useful when working with multiple user-defined evaluators that don't conform to the Patronus structured evaluator format.

```python
import patronus
from patronus.evals import bundled_eval, evaluator

patronus.init()

@evaluator()
def exact_match(actual, expected) -> bool:
    return actual == expected

@evaluator()
def iexact_match(actual: str, expected: str) -> bool:
    return actual.strip().lower() == expected.strip().lower()

# Group these evaluations together in a single trace and single log record
with bundled_eval():
    exact_match("string", "string")
    iexact_match("string", "string")
```
