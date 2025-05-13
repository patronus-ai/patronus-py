# Examples

Examples of how to use Patronus and what it can do.

## Usage

These examples demonstrate common use cases and integration patterns for Patronus.

### Setting required environment variables

Most examples require you to set up authentication with Patronus and other services. In most cases, you'll need to set the following environment variables:

```bash
export PATRONUS_API_KEY=your-api-key
export OPENAI_API_KEY=your-api-key
```

Some examples may require additional API keys (like `ANTHROPIC_API_KEY`).

### Running Examples

There are three ways to run the examples:

#### 1. Running with `uv`

You can run examples with `uv`, which automatically installs the required dependencies:

```bash
# Remember to export environment variables before running the example.
uv run --no-cache --with "patronus-examples[smolagents]" \
    -m patronus_examples.tracking.smolagents_weather
```

This installs the `patronus-examples` package with the necessary optional dependencies.

#### 2. Pulling the repository and executing the scripts directly

You can clone the repository and run the scripts directly:

```bash
# Clone the repository
git clone https://github.com/patronus-ai/patronus-py.git
cd patronus-py

# Run the example script (requires uv)
./examples/patronus_examples/tracking/smolagents_weather.py
```

See the script files for more information. They use uv script annotations to handle dependencies.

#### 3. Copy and paste example

You can copy the example code into your own project and install the dependencies with any package manager of your choice. Each example file includes a list of required dependencies at the top of the document.

## Available Examples

Patronus provides examples for various LLM frameworks and direct API integrations:

### Direct LLM API Integrations

- [OpenAI Weather Example](openai-weather.md) - Simple example of tracing OpenAI API calls
- [Anthropic Weather Example](anthropic-weather.md) - Simple example of tracing Anthropic API calls

### Agent Frameworks

- [Smolagents Weather](smolagents-weather.md) - Using Patronus with Smolagents
- [PydanticAI Weather](pydanticai-weather.md) - Using Patronus with PydanticAI
- [OpenAI Agents Weather](openai-agents-weather.md) - Using Patronus with OpenAI Agents
- [LangChain Weather](langchain-weather.md) - Using Patronus with LangChain and LangGraph
- [CrewAI Weather](crewai-weather.md) - Using Patronus with CrewAI

Each example demonstrates:
- How to set up Patronus integrations with the specific framework
- How to trace LLM calls and tool usage
- How to analyze the execution flow of your application

All examples follow a similar pattern using a weather application to make it easy to compare the different frameworks.

### Advanced Examples

- [Manual OpenTelemetry with OpenAI](otel-openai-weather.md) - An example showing how to use OpenTelemetry directly without Patronus SDK
