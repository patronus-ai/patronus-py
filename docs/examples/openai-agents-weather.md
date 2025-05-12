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
uv run --no-cache --with "patronus-examples[openai-agents]" \
    -m patronus_examples.tracking.openai_agents_weather
```

### Running the script directly

If you've cloned the repository, you can run the script directly:

```shell
# Clone the repository
git clone https://github.com/patronus-ai/patronus-py.git
cd patronus-py

# Run the example script (requires uv)
./examples/patronus_examples/tracking/openai_agents_weather.py
```

### Manual installation

If you prefer to copy the example code to your own project, you'll need to install these dependencies:

```shell
pip install patronus
pip install openai-agents
pip install openinference-instrumentation-openai-agents
pip install opentelemetry-instrumentation-threading
pip install opentelemetry-instrumentation-asyncio
```

## Example overview

This example demonstrates how to use Patronus to trace and monitor OpenAI Agents in an asynchronous weather application. The example:

1. Sets up a weather agent with a function tool to retrieve weather information
2. Creates a manager agent that can delegate to the weather agent
3. Handles the workflow using the OpenAI Agents Runner
4. Traces the entire agent execution flow with Patronus

The example shows how Patronus integrates with OpenAI Agents to provide visibility into agent hierarchies, tool usage, and asynchronous workflows.

## Example code

```python
# examples/patronus_examples/tracking/openai_agents_weather.py

from agents import Agent, Runner, function_tool
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
import asyncio
import patronus

patronus.init(
    integrations=[
        OpenAIAgentsInstrumentor(),
        ThreadingInstrumentor(),
        AsyncioInstrumentor(),
    ]
)


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"


def get_agents(tools=[]):
    weather_agent = Agent(
        name="weather_agent",
        instructions="You are a helpful assistant that can call tools and return weather related information",
        model="o3-mini",
        tools=tools,
    )

    manager_agent = Agent(
        name="manager_agent",
        instructions="You are a helpful assistant that can call other agents to accomplish different tasks",
        model="o3-mini",
        handoffs=[weather_agent],
    )
    return manager_agent


@patronus.traced("weather-openai-agent")
async def main():
    manager_agent = get_agents([get_weather])
    result = await Runner.run(manager_agent, "How is the weather in Paris, France?")
    return result.final_output


if __name__ == "__main__":
    print("Starting agent...")
    result = asyncio.run(main())
    print(result)
```
