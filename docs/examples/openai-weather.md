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
uv run --no-cache --with "patronus-examples[openai]" \
    -m patronus_examples.tracking.openai_weather
```

### Running the script directly

If you've cloned the repository, you can run the script directly:

```shell
# Clone the repository
git clone https://github.com/patronus-ai/patronus-py.git
cd patronus-py

# Run the example script (requires uv)
./examples/patronus_examples/tracking/openai_weather.py
```

### Manual installation

If you prefer to copy the example code to your own project, you'll need to install these dependencies:

```shell
pip install patronus
pip install openai
pip install openinference-instrumentation-openai
```

## Example overview

This example demonstrates how to use Patronus to trace OpenAI API calls when implementing a simple weather application. The application:

1. Uses the OpenAI API to parse a user question about weather
2. Extracts location coordinates from the LLM's output
3. Calls a weather API to get actual temperature data
4. Returns the result to the user

The example shows how Patronus can help you monitor and debug OpenAI API interactions, track tool usage, and visualize the entire application flow.

## Example code

```python
# examples/patronus_examples/tracking/openai_weather.py

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
```
