# Tracing

Tracing is a core feature of the Patronus SDK that allows you to monitor and understand the behavior of your LLM applications.
This page covers how to set up and use tracing in your code.

## Getting Started with Tracing

Tracing in Patronus works through two main mechanisms:

1. **Function decorators**: Easily trace entire functions
2. **Context managers**: Trace specific code blocks within functions

## Using the `@traced()` Decorator

The simplest way to add tracing is with the `@traced()` decorator:

```python
import patronus
from patronus import traced

patronus.init()

@traced()
def generate_response(prompt: str) -> str:
    # Your LLM call or processing logic here
    return f"Response to: {prompt}"

# Call the traced function
result = generate_response("Tell me about machine learning")
```

### Decorator Options

The `@traced()` decorator accepts several parameters for customization:

```python
@traced(
    span_name="Custom span name",   # Default: function name
    log_args=True,                  # Whether to log function arguments
    log_results=True,               # Whether to log function return values
    log_exceptions=True,            # Whether to log exceptions
    disable_log=False,              # Completely disable logging (maintains spans)
    attributes={"key": "value"}     # Custom attributes to add to the span
)
def my_function():
    pass
```

See the [API Reference][patronus.tracing.decorators.traced] for complete details.

## Using the `start_span()` Context Manager

For more granular control, use the `start_span()` context manager to trace specific blocks of code:

```python
import patronus
from patronus.tracing import start_span

patronus.init()

def complex_workflow(data):
    # First phase
    with start_span("Data preparation", attributes={"data_size": len(data)}):
        prepared_data = preprocess(data)

    # Second phase
    with start_span("Model inference"):
        results = run_model(prepared_data)

    # Third phase
    with start_span("Post-processing"):
        final_results = postprocess(results)

    return final_results
```

### Context Manager Options

The `start_span()` context manager accepts these parameters:

```python
with start_span(
    "Span name",                        # Name of the span (required)
    record_exception=False,             # Whether to record exceptions
    attributes={"custom": "attribute"}  # Custom attributes to add
) as span:
    # Your code here
    # You can also add attributes during execution:
    span.set_attribute("dynamic_value", 42)
```

See the [API Reference][patronus.tracing.decorators.start_span] for complete details.

## Custom Attributes

Both tracing methods allow you to add custom attributes that provide additional context for your traces:

```python
@traced(attributes={
    "model": "gpt-4",
    "version": "1.0",
    "temperature": 0.7
})
def generate_with_gpt4(prompt):
    # Function implementation
    pass

# Or with context manager
with start_span("Query processing", attributes={
    "query_type": "search",
    "filters_applied": True,
    "result_limit": 10
}):
    # Processing code
    pass
```
