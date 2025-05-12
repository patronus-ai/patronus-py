#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "patronus",
#     "openai-agents",
#     "openinference-instrumentation-openai-agents>=0.1.11,<0.2.0",
#     "opentelemetry-instrumentation-threading>=0.54b0,<1.0",
#     "opentelemetry-instrumentation-asyncio>=0.54b0,<1.0"
# ]
# ///
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
