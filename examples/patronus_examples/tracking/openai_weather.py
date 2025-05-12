#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "patronus",
#     "openai",
#     "openinference-instrumentation-openai>=0.1.28,<0.2.0",
#     "opentelemetry-instrumentation-threading>=0.54b0,<1.0",
#     "opentelemetry-instrumentation-asyncio>=0.54b0,<1.0"
# ]
# ///
import json

import requests
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
import patronus

# Initialize patronus with OpenAI Instrumentor
patronus.init(integrations=[OpenAIInstrumentor()])


def get_weather(latitude, longitude):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]["temperature_2m"]


def get_client():
    client = OpenAI()
    return client


@patronus.traced()
def call_llm(client, user_prompt):
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    input_messages = [{"role": "user", "content": user_prompt}]

    response = client.responses.create(
        model="gpt-4.1",
        input=input_messages,
        tools=tools,
    )
    return response


@patronus.traced("openai-weather")
def main():
    user_prompt = "What's the weather like in Paris today?"

    client = get_client()
    response = call_llm(client, user_prompt)
    print("LLM Response")
    print(response.model_dump_json())

    weather_response = None
    if response.output:
        output = response.output[0]
        if output.type == "function_call" and output.name == "get_weather":
            kwargs = json.loads(output.arguments)
            print("Weather API Response")
            weather_response = get_weather(**kwargs)
            print(weather_response)

    if weather_response:
        print(user_prompt)
        print(f"Answer: {weather_response}")


if __name__ == "__main__":
    main()
