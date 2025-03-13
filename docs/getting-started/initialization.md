
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

## Next Steps

Now that you've installed the Patronus SDK, proceed to the [Quickstart](quickstart.md) guide to learn how to use it effectively.
