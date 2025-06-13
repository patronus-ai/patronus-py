
## API Key

To use the Patronus SDK, you'll need an API key from the Patronus platform. If you don't have one yet:

1. Sign up at [https://app.patronus.ai](https://app.patronus.ai)
2. Navigate to "API Keys"
3. Create a new API key

## Configuration

There are several ways to configure the Patronus SDK:

### Environment Variables

Set your API key as an environment variable:

```bash
export PATRONUS_API_KEY="your-api-key"
```

### Configuration File

Create a `patronus.yaml` file in your project directory:

```yaml
api_key: "your-api-key"
project_name: "Global"
app: "default"
```

### Direct Configuration

Pass configuration values directly when initializing the SDK:

```python
import patronus

patronus.init(
    api_key="your-api-key",
    project_name="Global",
    app="default",
)
```

## Verification

To verify your installation and configuration:

```python
import patronus

patronus.init()

# Create a simple tracer
@patronus.traced()
def test_function():
    return "Installation successful!"

# Call the function to test tracing
result = test_function()
print(result)
```

If no errors occur, your Patronus SDK is correctly installed and configured.

## Advanced

### Return Value

The `patronus.init()` function returns a [`PatronusContext`][patronus.context.PatronusContext] object that serves as the central access point for all SDK components and functionality. Additionally, `patronus.init()` automatically sets this context globally, making it accessible throughout your application:

```python
import patronus

# Capture the returned context
patronus_context = patronus.init()  # Also sets context globally

# Direct access is possible but not typically needed
tracer_provider = patronus_context.tracer_provider
api_client = patronus_context.api_client
scope = patronus_context.scope
```

See the [`PatronusContext`][patronus.context.PatronusContext] API reference for the complete list of available components and their descriptions.

This context is particularly useful when integrating with OpenTelemetry instrumentation libraries that require explicit tracer provider configuration, such as in [distributed tracing scenarios](../observability/tracing.md#distributed-tracing).

### Manual Context Management

For advanced use cases, you can build and manage contexts manually using [`build_context()`][patronus.init.build_context] and the context manager pattern:

```python
from patronus.init import build_context
from patronus import context

# Build a context manually with custom configuration
custom_context = build_context(...)

# Use the context temporarily without setting it globally
with context._CTX_PAT.using(custom_context):
    # All Patronus SDK operations within this block use custom_context
    result = some_patronus_operation()
# Context reverts to previous state after exiting the block
```

This pattern is particularly useful when you need to send data to multiple projects within the same process, or when building testing frameworks that require isolated contexts.

## Next Steps

Now that you've installed the Patronus SDK, proceed to the [Quickstart](quickstart.md) guide to learn how to use it effectively.
