# LLM Integrations

The Patronus SDK provides integrations with various LLM providers to enable observability, evaluation, and experimentation with LLM applications.

## OpenTelemetry LLM Instrumentors

Patronus supports any OpenTelemetry-based LLM instrumentation. This allows you to easily capture telemetry data from your LLM interactions and send it to the Patronus platform for analysis.

A popular option for LLM instrumentation is [OpenInference](https://github.com/Arize-ai/openinference), which provides instrumentors for multiple LLM providers.

### Anthropic Claude Integration

To instrument Anthropic's Claude API calls:

```shell
# Install the required package
pip install openinference-instrumentation-anthropic
```

```python
from patronus import init
from openinference.instrumentation.anthropic import AnthropicInstrumentor

# Initialize Patronus with Anthropic instrumentation
patronus_ctx = init(
    integrations=[AnthropicInstrumentor()]
)

# Now all Claude API calls will be automatically instrumented
# and the telemetry will be sent to Patronus
```

### OpenAI Integration

To instrument OpenAI API calls:

```shell
# Install the required package
pip install openinference-instrumentation-openai
```

```python

from patronus import init
from openinference.instrumentation.openai import OpenAIInstrumentor

# Initialize Patronus with OpenAI instrumentation
patronus_ctx = init(
    integrations=[OpenAIInstrumentor()]
)

# Now all OpenAI API calls will be automatically instrumented
# and the telemetry will be sent to Patronus
```

### Using Multiple LLM Instrumentors

You can combine multiple instrumentors to capture telemetry from different LLM providers:

```python
from patronus import init
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor

# Initialize Patronus with multiple LLM instrumentors
patronus_ctx = init(
    project_name="my-multi-llm-project",
    app="llm-application",
    integrations=[
        AnthropicInstrumentor(),
        OpenAIInstrumentor()
    ]
)

# Now both Anthropic and OpenAI API calls will be automatically instrumented
```
