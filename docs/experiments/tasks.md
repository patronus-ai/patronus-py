# Creating Tasks

Tasks in Patronus experiments are functions that process each dataset example and produce outputs that will be evaluated.
This page covers how to create and use tasks effectively.

## Task Function Basics

A task function receives a dataset row and produces an output. The simplest task functions look like this:

```python
def simple_task(row, **kwargs):
    # Process the input from the row
    input_text = row.task_input

    # Generate an output (typically a score between 0 and 1)
    quality_score = 0.85

    # Return the output as a float
    return quality_score
```

The framework automatically converts numeric outputs to `TaskResult` objects.

## Task Function Parameters

Task functions always receive these parameters:

- `row`: [Row][patronus.datasets.datasets.Row] - The dataset example to process
- `parent`: [EvalParent][patronus.experiments.types.EvalParent] - Information from previous chain stages (if any)
- `tags`: [Tags][patronus.experiments.experiment.Tags] - Tags associated with the experiment and dataset
- `**kwargs`: Additional keyword arguments

Here's a more complete task function:

```python
from patronus.datasets import Row
from patronus.experiments.types import EvalParent

def complete_task(
    row: Row,
    parent: EvalParent = None,
    tags: dict[str, str] = None,
    **kwargs
):
    # Access dataset fields
    input_text = row.task_input
    context = row.task_context
    system_prompt = row.system_prompt
    gold_answer = row.gold_answer

    # Access parent information (from previous chain steps)
    previous_output = None
    if parent and parent.task:
        previous_output = parent.task.output

    # Access tags
    model_name = tags.get("model_name", "default")

    # Generate output (in real usage, this would call an LLM)
    output = f"Model {model_name} processed: {input_text}"

    # Return the output
    return output
```

## Return Types

Task functions can return several types:

### String Output

Here's an improved example for the string return type section that demonstrates a classification task:

```python
def classify_sentiment(row: Row, **kwargs) -> str:
    # Extract the text to classify
    text = row.task_input

    # Simple rule-based sentiment classifier
    positive_words = ["good", "great", "excellent", "happy", "positive"]
    negative_words = ["bad", "terrible", "awful", "sad", "negative"]

    text_lower = text.lower()
    positive_count = sum(word in text_lower for word in positive_words)
    negative_count = sum(word in text_lower for word in negative_words)

    # Classify based on word counts
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"
```

The string output represents a specific classification category, which is a common pattern in text classification tasks.

### Numeric Output (Float/Int)

For score-based outputs:

```python
def score_task(row: Row, **kwargs) -> float:
    # Calculate a relevance score between 0 and 1
    return 0.92
```

### TaskResult Object

For more control, return a [TaskResult][patronus.experiments.types.TaskResult] object:

```python
from patronus.experiments.types import TaskResult

def task_result(row: Row, **kwargs) -> TaskResult:
    # Generate output
    output = f"Processed: {row.task_input}"

    # Include metadata about the processing
    metadata = {
        "processing_time_ms": 42,
        "confidence": 0.95,
        "tokens_used": 150
    }

    # Add tags for filtering and organization
    tags = {
        "model": "gpt-4",
        "temperature": "0.7"
    }

    # Return a complete TaskResult
    return TaskResult(
        output=output,
        metadata=metadata,
        tags=tags
    )
```

### None / Skipping Examples

Return `None` to skip processing this example:

```python
def selective_task(row: Row, **kwargs) -> None:
    # Skip examples without the required fields
    if not row.task_input or not row.gold_answer:
        return None

    # Process valid examples
    return f"Processed: {row.task_input}"
```

## Calling LLMs

A common use of tasks is to generate outputs using Large Language Models:

```python
from openai import OpenAI
from patronus.datasets import Row
from patronus.experiments.types import TaskResult

oai = OpenAI()

def openai_task(row: Row, **kwargs) -> TaskResult:
    # Prepare the input for the model
    system_message = row.system_prompt or "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": row.task_input}
    ]

    # Call the OpenAI API
    response = oai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )

    # Extract the output
    output = response.choices[0].message.content

    # Include metadata about the call
    metadata = {
        "model": response.model,
        "tokens": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }
    }

    return TaskResult(
        output=output,
        metadata=metadata
    )
```

## Async Tasks

For better performance, especially with API calls, you can use async tasks:

```python
import asyncio
from openai import AsyncOpenAI
from patronus.datasets import Row
from patronus.experiments.types import TaskResult

oai = AsyncOpenAI()

async def async_openai_task(
    row: Row,
    parent: EvalParent = None,
    tags: dict[str, str] = None,
    **kwargs
) -> TaskResult:
    # Create async client

    # Prepare the input
    system_message = row.system_prompt or "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": row.task_input}
    ]

    # Call the OpenAI API asynchronously
    response = await oai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )

    # Extract and return the output
    output = response.choices[0].message.content

    return TaskResult(
        output=output,
        metadata={"model": response.model}
    )
```

The Patronus framework automatically handles both synchronous and asynchronous tasks.

## Using Parent Information

In multi-stage chains, tasks can access the results of previous stages:

```python
from patronus.datasets import Row
from patronus.experiments.types import EvalParent

def second_stage_task(
    row: Row,
    parent: EvalParent,
    tags: dict[str, str] = None,
    **kwargs
) -> str:
    # Access previous task output
    if parent and parent.task:
        previous_output = parent.task.output
        return f"Building on previous output: {previous_output}"

    # Fallback if no previous output
    return f"Starting fresh: {row.task_input}"
```

## Error Handling

Task functions should handle exceptions appropriately:

```python
from patronus import get_logger
from patronus.datasets import Row

def robust_task(row: Row, **kwargs):
    try:
        # Attempt to process
        if row.task_input:
            return f"Processed: {row.task_input}"
        else:
            # Skip if input is missing
            return None
    except Exception as e:
        # Log the error
        get_logger().exception(f"Error processing row {row.sid}: {e}")
        # Skip this example
        return None
```

If an unhandled exception occurs, the experiment will log the error and skip that example.

## Task Tracing

Tasks are automatically traced with the Patronus tracing system. You can add additional tracing:

```python
from patronus.tracing import start_span
from patronus.datasets import Row

def traced_task(row: Row, **kwargs):
    # Outer span is created automatically by the framework

    # Create spans for subtasks
    with start_span("Preprocessing"):
        # Preprocessing logic...
        preprocessed = preprocess(row.task_input)

    with start_span("Model Call"):
        # Model call logic...
        output = call_model(preprocessed)

    with start_span("Postprocessing"):
        # Postprocessing logic...
        final_output = postprocess(output)

    return final_output
```

This helps with debugging and performance analysis.

## Best Practices

When creating task functions:

1. **Handle missing data gracefully**: Check for required fields and handle missing data
2. **Include useful metadata**: Add information about processing steps, model parameters, etc.
3. **Use async for API calls**: Async tasks significantly improve performance for API-dependent workflows
4. **Add explanatory tags**: Tags help with filtering and analyzing results
5. **Add tracing spans**: For complex processing, add spans to help with debugging and optimization
6. **Keep functions focused**: Tasks should have a clear purpose; use chains for multi-step processes

Next, we'll explore how to use evaluators in experiments to assess task outputs.
