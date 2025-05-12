#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "patronus",
#     "anthropic",
#     "openinference-instrumentation-anthropic>=0.1.17,<0.2.0",
#     "opentelemetry-instrumentation-threading>=0.54b0,<1.0",
#     "opentelemetry-instrumentation-asyncio>=0.54b0,<1.0"
# ]
# ///
import requests

import anthropic
from openinference.instrumentation.anthropic import AnthropicInstrumentor
import patronus

# Initialize patronus with Anthropic Instrumentor
patronus.init(integrations=[AnthropicInstrumentor()])


def get_weather(latitude, longitude):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]["temperature_2m"]


def get_client():
    client = anthropic.Anthropic()
    return client


@patronus.traced()
def call_llm(client, user_prompt):
    tools = [
        {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        }
    ]

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response


@patronus.traced("anthropic-weather")
def main():
    user_prompt = "What's the weather like in Paris today?"

    client = get_client()
    response = call_llm(client, user_prompt)
    print("LLM Response")
    print(response.model_dump_json())

    weather_response = None
    if response.content:
        for content_block in response.content:
            if content_block.type == "tool_use" and content_block.name == "get_weather":
                kwargs = content_block.input
                print("Weather API Response")
                weather_response = get_weather(**kwargs)
                print(weather_response)

    if weather_response:
        print(user_prompt)
        print(f"Answer: {weather_response}")


if __name__ == "__main__":
    main()
