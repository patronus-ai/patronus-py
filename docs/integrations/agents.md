# Agent Integrations

The Patronus SDK provides integrations with various agent frameworks to enable observability, evaluation, and experimentation with agent-based LLM applications.

## Pydantic AI

[Pydantic AI](https://ai.pydantic.dev/) is a framework for building AI agents with type-safe tools and structured outputs. The Patronus SDK provides a dedicated integration that automatically instruments all Pydantic AI agents for observability.

### Installation

Make sure you have both the Patronus SDK and Pydantic AI installed:

```bash
pip install patronus pydantic-ai
```

### Usage

To enable Pydantic AI integration with Patronus:

```python
from patronus import init
from patronus.integrations.pydantic_ai import PydanticAIIntegrator

# Initialize Patronus with the Pydantic AI integration
patronus_ctx = init(
    integrations=[PydanticAIIntegrator()]
)

# Now all Pydantic AI agents will automatically send telemetry to Patronus
```

### Configuration Options

The `PydanticAIIntegrator` accepts the following parameters:

- `event_mode`: Controls how agent events are captured
    - `"logs"` (default): Captures events as logs, which works best with the Patronus Platform
    - `"attributes"`: Captures events as span attributes

Example with custom configuration:

```python
from patronus import init
from patronus.integrations.pydantic_ai import PydanticAIIntegrator

patronus_ctx = init(
    integrations=[PydanticAIIntegrator(event_mode="logs")]
)
```
