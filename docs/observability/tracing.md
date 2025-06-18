# Tracing

Tracing is a core feature of the Patronus SDK that allows you to monitor and understand the behavior of your LLM applications.
This page covers how to set up and use tracing in your code.

!!! tip "Configuration"

    For information about configuring observability features, including exporter protocols and endpoints, see the [Observability Configuration](configuration.md) guide.

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

## Distributed Tracing

The Patronus SDK is built on OpenTelemetry and automatically supports context propagation across distributed services. This enables you to trace requests as they flow through multiple services in your application architecture. The [OpenTelemetry Python Contrib](https://github.com/open-telemetry/opentelemetry-python-contrib) repository provides instrumentation for many popular frameworks and libraries.

### Example: FastAPI Services with Context Propagation

First, install the required dependencies:

```bash
uv add opentelemetry-instrumentation-httpx \
    opentelemetry-instrumentation-fastapi \
    fastapi[all] \
    patronus
```

Here's a complete example showing two FastAPI services with automatic trace context propagation:

**Backend Service (`service_backend.py`):**

```python
import patronus
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize Patronus SDK
patronus_context = patronus.init(service="backend")

app = FastAPI(title="Backend Service")

@app.get("/hello/{name}")
async def hello_backend(name: str):
    return {
        "message": f"Hello {name} from Backend Service!",
        "service": "backend"
    }

# Instrument FastAPI after Patronus initialization
FastAPIInstrumentor.instrument_app(app, tracer_provider=patronus_context.tracer_provider)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Gateway Service (`service_gateway.py`):**

```python
import httpx
import patronus
from fastapi import FastAPI
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize Patronus SDK with HTTPX instrumentation
patronus_context = patronus.init(
    service="gateway",
    integrations=[
        HTTPXClientInstrumentor(),
    ]
)

app = FastAPI(title="Gateway Service")

@app.get("/hello/{name}")
async def hello_gateway(name: str):
    # This HTTP call will automatically propagate trace context
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://localhost:8001/hello/{name}")
        backend_data = response.json()

    return {
        "gateway_message": f"Gateway received request for {name}",
        "backend_response": backend_data
    }

# Instrument FastAPI after Patronus initialization
FastAPIInstrumentor.instrument_app(app, tracer_provider=patronus_context.tracer_provider)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Running the Example

First, export your Patronus API key:

```bash
export PATRONUS_API_KEY="your-api-key"
```

Then run the services:

1. Start the backend: `python service_backend.py`
2. Start the gateway: `python service_gateway.py`
3. Make a request: `curl http://localhost:8000/hello/world`

After making the request, you should see the connected traces in the Patronus Platform showing the complete request flow from gateway to backend service.

### Important Notes

- FastAPI instrumenter requires manual setup with `FastAPIInstrumentor.instrument_app()` after Patronus initialization
- Pass the `tracer_provider` from Patronus context to ensure proper integration
- Trace context is automatically propagated through HTTP headers when services are properly instrumented
