#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "patronus",
#     "openinference-instrumentation-smolagents>=0.1.11,<0.2.0",
#     "opentelemetry-instrumentation-asyncio",
#     "opentelemetry-instrumentation-threading",
#     "smolagents[litellm]",
# ]
# ///
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
