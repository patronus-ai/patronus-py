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
