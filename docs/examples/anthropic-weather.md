## Running the example

To run this example, you need to add API keys to your environment:

```shell
export PATRONUS_API_KEY=your-api-key
export ANTHROPIC_API_KEY=your-api-key
```

### Running with `uv`

You can run the example as a one-liner with zero setup:

```shell
# Remember to export environment variables before running the example.
uv run --no-cache --with "patronus-examples[anthropic]" \
    -m patronus_examples.tracking.anthropic_weather
```

### Running the script directly

If you've cloned the repository, you can run the script directly:

```shell
# Clone the repository
git clone https://github.com/patronus-ai/patronus-py.git
cd patronus-py

# Run the example script (requires uv)
./examples/patronus_examples/tracking/anthropic_weather.py
```

### Manual installation

If you prefer to copy the example code to your own project, you'll need to install these dependencies:

```shell
pip install patronus
pip install anthropic
pip install openinference-instrumentation-anthropic
```

## Example overview

This example demonstrates how to use Patronus to trace Anthropic API calls when implementing a simple weather application. The application:

1. Uses the Anthropic Claude API to parse a user question about weather
2. Extracts location coordinates from the LLM's output through Claude's tool calling
3. Calls a weather API to get actual temperature data
4. Returns the result to the user

The example shows how Patronus can help you monitor and debug Anthropic API interactions, track tool usage, and visualize the entire application flow.

## Example code

```python
# examples/patronus_examples/tracking/anthropic_weather.py

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
```
