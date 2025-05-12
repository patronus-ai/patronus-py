## Running the example

To run this example, you need to add API keys to your environment:

```shell
export PATRONUS_API_KEY=your-api-key
export OPENAI_API_KEY=your-api-key
```

### Running with `uv`

You can run the example as a one-liner with zero setup:

```shell
# Remember to export environment variables before running the example.
uv run --no-cache --with "patronus-examples[pydantic-ai]" \
    -m patronus_examples.tracking.pydanticai_weather
```

### Running the script directly

If you've cloned the repository, you can run the script directly:

```shell
# Clone the repository
git clone https://github.com/patronus-ai/patronus-py.git
cd patronus-py

# Run the example script (requires uv)
./examples/patronus_examples/tracking/pydanticai_weather.py
```

### Manual installation

If you prefer to copy the example code to your own project, you'll need to install these dependencies:

```shell
pip install patronus
pip install pydantic-ai-slim[openai]
pip install opentelemetry-instrumentation-asyncio
pip install opentelemetry-instrumentation-threading
```

## Example overview

This example demonstrates how to use Patronus to trace Pydantic-AI agent interactions in an asynchronous application. The example:

1. Sets up two Pydantic-AI agents: a weather agent and a manager agent
2. Configures the weather agent with a tool to provide mock weather data
3. Configures the manager agent with a tool to call the weather agent
4. Demonstrates how to handle agent-to-agent communication

The example shows how Patronus can trace asynchronous workflows and provide visibility into multi-agent systems built with Pydantic-AI.

## Example code

```python
# examples/patronus_examples/tracking/pydanticai_weather.py

import asyncio
from pydantic_ai import Agent
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from patronus.integrations.pydantic_ai import PydanticAIIntegrator
import patronus

patronus.init(
    integrations=[
        PydanticAIIntegrator(),
        ThreadingInstrumentor(),
        AsyncioInstrumentor(),
    ]
)


def get_agent(system_prompt="You are a helpful assistant"):
    agent = Agent("openai:gpt-4o", output_type=str, system_prompt=system_prompt)
    return agent


@patronus.traced("weather-pydantic-ai")
async def main():
    # Create weather agent and attach tool to it
    weather_agent = get_agent(
        "You are a helpful assistant that can help with weather information."
    )

    @weather_agent.tool_plain()
    async def get_weather():
        # Mock tool output
        return (
            "Today's weather is Sunny with a forecasted high of 30°C and a low of 25°C. "
            "The wind is expected at 4 km/h."
        )

    # Create manager agent
    manager_agent = get_agent(
        "You are a helpful assistant that can coordinate with other subagents "
        "and query them for more information about topics."
    )

    # Create a tool to execute the weather agent
    @manager_agent.tool_plain()
    async def call_weather_agent():
        weather_info = await weather_agent.run("What is the weather in Paris, France?")
        return str(weather_info)

    # Run the manager
    print("Running the agent...")
    return await manager_agent.run("What is the weather in Paris, France?")


if __name__ == "__main__":
    result = asyncio.run(main())
    print(result)
```
