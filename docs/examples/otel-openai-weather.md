## Manual OpenTelemetry Tracing Example

This example demonstrates how to use OpenTelemetry (OTel) directly with OpenInference instrumenters to trace a simple OpenAI weather application **without** using Patronus SDK. This shows how to implement manual instrumentation combined with automatic instrumenters.

## Running the example

To run this example, you need to add your OpenAI API key to your environment:

```shell
export OPENAI_API_KEY=your-api-key
```

### Running with `uv`

You can run the example as a one-liner with zero setup:

```shell
# Remember to export environment variables before running the example
uv run --no-cache --with "patronus-examples opentelemetry-api>=1.31.0 opentelemetry-sdk>=1.31.0 opentelemetry-exporter-otlp>=1.31.0 openinference-instrumentation-openai>=0.1.28 openai httpx>=0.27.0" \
    -m patronus_examples.tracking.otel_openai_weather
```

### Running with Patronus OTel collector

To export traces to Patronus OTel collector, set these additional environment variables:

```shell
export PATRONUS_API_KEY=your-api-key
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otel.patronus.ai:4317"
export OTEL_EXPORTER_OTLP_HEADERS="x-api-key=$PATRONUS_API_KEY"
```

### Manual installation

If you prefer to copy the example code to your own project, you'll need to install these dependencies:

```shell
pip install openai
pip install opentelemetry-api
pip install opentelemetry-sdk
pip install opentelemetry-exporter-otlp
pip install openinference-instrumentation-openai
pip install httpx
```

## Example overview

This example demonstrates how to combine manual OpenTelemetry instrumentation with OpenInference auto-instrumentation for an OpenAI-based weather application. The application:

1. Sets up a complete OpenTelemetry tracing pipeline
2. Initializes OpenInference instrumenter for OpenAI
3. Calls the OpenAI API which is automatically traced by OpenInference
4. Adds additional manual spans for non-OpenAI components
5. Makes an instrumented HTTP request using httpx to a weather API
6. Records all relevant attributes and events in spans

The example shows how to:

- Configure an OpenTelemetry TracerProvider
- Set up either console or OTLP exporters
- Initialize OpenInference instrumenters with OpenTelemetry
- Create nested manual spans for tracking operations
- Use httpx for HTTP requests with proper tracing
- Add attributes to spans for better observability
- Handle errors and exceptions in spans

## Example code

```python
# examples/patronus_examples/tracking/otel_openai_weather.py

import json
import os
import httpx
from openai import OpenAI

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

# Import OpenInference instrumenter for OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor

# Configure OpenTelemetry
resource = Resource(attributes={
    ResourceAttributes.SERVICE_NAME: "openai-weather-app",
    ResourceAttributes.SERVICE_VERSION: "0.1.0",
})

# Initialize the trace provider with the resource
trace_provider = TracerProvider(resource=resource)

# If OTEL_EXPORTER_OTLP_ENDPOINT is not set, we'll use console exporter
# Otherwise, we'll use OTLP exporter for sending to the Patronus collector
if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    # Configure OTLPSpanExporter
    # The environment variables OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS
    # should be set before running this example
    otlp_exporter = OTLPSpanExporter()
    trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
else:
    # For local development/testing we can use ConsoleSpanExporter
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

# Set the provider
trace.set_tracer_provider(trace_provider)

# Initialize OpenInference instrumenter for OpenAI
# This will automatically instrument all OpenAI API calls
openai_instrumentor = OpenAIInstrumentor()
openai_instrumentor.instrument()

# Get a tracer for our manual spans
tracer = trace.get_tracer("openai.weather.example")


def get_weather(latitude, longitude):
    """Get weather data from the Open Meteo API using httpx"""
    with tracer.start_as_current_span(
        "get_weather",
        attributes={
            "service.name": "weather_api",
            "weather.latitude": latitude,
            "weather.longitude": longitude
        }
    ) as span:
        try:
            # Create the URL with parameters
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m",
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"
            }

            # Trace the HTTP request using httpx
            with tracer.start_as_current_span(
                "http_request",
                attributes={
                    "http.method": "GET",
                    "http.url": url,
                    "http.request.query": str(params)
                }
            ):
                # Use httpx client for the request
                with httpx.Client() as client:
                    response = client.get(url, params=params)

                # Add response information to the span
                span.set_attribute("http.status_code", response.status_code)

                if response.status_code != 200:
                    span.record_exception(Exception(f"Weather API returned status {response.status_code}"))
                    span.set_status(trace.StatusCode.ERROR)
                    return None

                data = response.json()
                temperature = data["current"]["temperature_2m"]

                # Add weather data to the span
                span.set_attribute("weather.temperature_celsius", temperature)

                return temperature
        except Exception as e:
            # Record the exception in the span
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise


def get_client():
    """Create and return an OpenAI client"""
    with tracer.start_as_current_span("get_openai_client"):
        return OpenAI()


def call_llm(client, user_prompt):
    """Call the OpenAI API to process the user prompt

    Note: With OpenInference instrumenter, the OpenAI API call will be
    automatically traced. This function adds some additional manual spans
    for demonstration purposes.
    """
    with tracer.start_as_current_span(
        "call_llm",
        attributes={
            "ai.prompt.text": user_prompt,
            "ai.prompt.tokens": len(user_prompt.split())
        }
    ) as span:
        try:
            # Define tools available to the model
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

            # The OpenAI API call will be automatically traced by OpenInference
            # We don't need to create a span for it, but we can add attributes to our parent span
            response = client.responses.create(
                model="gpt-4.1",
                input=input_messages,
                tools=tools,
            )

            # Check if the response contains a tool call
            has_tool_call = False
            if response.output and len(response.output) > 0:
                output = response.output[0]
                if output.type == "function_call":
                    has_tool_call = True
                    span.set_attribute("openai.response.tool_called", output.name)

            span.set_attribute("openai.response.has_tool_call", has_tool_call)

            return response
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise


def main():
    """Main function to process the weather query"""
    with tracer.start_as_current_span("openai-weather-main") as root_span:
        user_prompt = "What's the weather like in Paris today?"
        root_span.set_attribute("query", user_prompt)

        try:
            client = get_client()
            response = call_llm(client, user_prompt)
            print("LLM Response")
            print(response.model_dump_json())

            weather_response = None
            if response.output:
                output = response.output[0]
                if output.type == "function_call" and output.name == "get_weather":
                    # Parse the arguments from the function call
                    with tracer.start_as_current_span(
                        "parse_function_call",
                        attributes={"function_name": output.name}
                    ):
                        kwargs = json.loads(output.arguments)
                        root_span.set_attribute("weather.latitude", kwargs.get("latitude"))
                        root_span.set_attribute("weather.longitude", kwargs.get("longitude"))

                    print("Weather API Response")
                    weather_response = get_weather(**kwargs)
                    print(weather_response)

            if weather_response:
                with tracer.start_as_current_span("format_weather_response"):
                    print(user_prompt)
                    formatted_answer = f"Answer: {weather_response}"
                    print(formatted_answer)
                    root_span.set_attribute("weather.answer", formatted_answer)

            # Mark the trace as successful
            root_span.set_status(trace.StatusCode.OK)

        except Exception as e:
            # Record any exceptions that occurred
            root_span.record_exception(e)
            root_span.set_status(trace.StatusCode.ERROR, str(e))
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
    # Ensure all spans are exported before the program exits
    trace_provider.shutdown()
```
