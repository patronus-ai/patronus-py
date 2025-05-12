#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "patronus",
#     "pydantic-ai-slim[openai]",
#     "opentelemetry-instrumentation-asyncio>=0.54b0,<1.0",
#     "opentelemetry-instrumentation-threading>=0.54b0,<1.0",
# ]
# ///
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
