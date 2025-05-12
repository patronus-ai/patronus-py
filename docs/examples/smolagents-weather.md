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
uv run --no-cache --with "patronus-examples[smolagents]" \
    -m patronus_examples.tracking.smolagents_weather
```

### Running the script directly

If you've cloned the repository, you can run the script directly:

```shell
# Clone the repository
git clone https://github.com/patronus-ai/patronus-py.git
cd patronus-py

# Run the example script (requires uv)
./examples/patronus_examples/tracking/smolagents_weather.py
```

### Manual installation

If you prefer to copy the example code to your own project, you'll need to install these dependencies:

```shell
pip install patronus
pip install smolagents[litellm]
pip install openinference-instrumentation-smolagents
pip install opentelemetry-instrumentation-threading
```

## Example overview

This example demonstrates how to use Patronus to trace Smolagents tool calls and LLM interactions. The application:

1. Sets up a Smolagents agent with a weather tool
2. Configures a hierarchical agent structure with subagents
3. Processes a user query about weather in Paris
4. Handles the tool calling workflow automatically

The example shows how Patronus provides visibility into the agent's decision-making process, tool usage, and interaction between different agent layers.

## Example code

```python
# examples/patronus_examples/tracking/smolagents_weather.py

from datetime import datetime

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from smolagents import LiteLLMModel, ToolCallingAgent, tool

import patronus

patronus.init(integrations=[SmolagentsInstrumentor(), ThreadingInstrumentor()])


@tool
def get_weather_api(location: str, date_time: str) -> str:
    """
    Returns the weather report.

    Args:
        location: the name of the place that you want the weather for.
            Should be a place name, followed by possibly a city name, then a country,
            like "Anchor Point, Taghazout, Morocco".
        date_time: the date and time for which you want the report, formatted as '%m/%d/%y %H:%M:%S'.
    """
    try:
        date_time = datetime.strptime(date_time, "%m/%d/%y %H:%M:%S")
    except Exception as e:
        raise ValueError(
            "Conversion of `date_time` to datetime format failed, "
            f"make sure to provide a string in format '%m/%d/%y %H:%M:%S': {e}"
        )
    temperature_celsius, risk_of_rain, wave_height = 10, 0.5, 4  # mock outputs
    return (
        f"Weather report for {location}, {date_time}: "
        f"Temperature will be {temperature_celsius}°C, "
        f"risk of rain is {risk_of_rain * 100:.0f}%, wave height is {wave_height}m."
    )


def create_agent(model_id):
    # Create weather agent
    weather_model = LiteLLMModel(model_id, temperature=0.0, top_p=1.0)
    weather_subagent = ToolCallingAgent(
        tools=[get_weather_api],
        model=weather_model,
        max_steps=10,
        name="weather_agent",
        description="This agent can provide information about the weather at a certain location",
    )

    # Create manager agent and add weather agent as subordinate
    manager_model = LiteLLMModel(model_id, temperature=0.0, top_p=1.0)
    agent = ToolCallingAgent(
        model=manager_model,
        managed_agents=[weather_subagent],
        tools=[],
        add_base_tools=False,
    )
    return agent


@patronus.traced("weather-smolagents")
def main():
    agent = create_agent("openai/gpt-4o")
    agent.run("What is the weather in Paris, France?")


if __name__ == "__main__":
    main()
```
