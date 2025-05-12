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
uv run --no-cache --with "patronus-examples[crewai]" \
    -m patronus_examples.tracking.crewai_weather
```

### Running the script directly

If you've cloned the repository, you can run the script directly:

```shell
# Clone the repository
git clone https://github.com/patronus-ai/patronus-py.git
cd patronus-py

# Run the example script (requires uv)
./examples/patronus_examples/tracking/crewai_weather.py
```

### Manual installation

If you prefer to copy the example code to your own project, you'll need to install these dependencies:

```shell
pip install patronus
pip install crewai
pip install openinference.instrumentation.crewai
pip install opentelemetry-instrumentation-threading
pip install opentelemetry-instrumentation-asyncio
```

## Example overview

This example demonstrates how to use Patronus to trace and monitor CrewAI agents in a weather application. The example:

1. Sets up a specialized Weather Information Specialist agent with a custom weather tool
2. Creates a manager agent that coordinates information requests
3. Defines tasks for each agent to perform
4. Configures a hierarchical workflow using the CrewAI Crew construct
5. Traces the entire execution flow with Patronus

The example shows how Patronus integrates with CrewAI to provide visibility into agent interactions, tool usage, and the hierarchical task execution process.

## Example code

```python
# examples/patronus_examples/tracking/crewai_weather.py

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from openinference.instrumentation.crewai import CrewAIInstrumentor
import patronus

patronus.init(
    integrations=[CrewAIInstrumentor(), ThreadingInstrumentor(), AsyncioInstrumentor()]
)


# Create a custom tool for weather information
class WeatherTool(BaseTool):
    name: str = "get_weather_api"
    description: str = "Returns the weather report for a specific location"

    def _run(self, location: str) -> str:
        """
        Returns the weather report.

        Args:
            location: the name of the place that you want the weather for. Should be a place name, followed by possibly a city name, then a country, like "Anchor Point, Taghazout, Morocco".

        Returns:
            The weather report.
        """
        temperature_celsius, risk_of_rain, wave_height = 10, 0.5, 4  # mock outputs
        return f"Weather report for {location}: Temperature will be {temperature_celsius}°C, risk of rain is {risk_of_rain * 100:.0f}%, wave height is {wave_height}m."


# Initialize weather tool
weather_tool = WeatherTool()

# Define agents
weather_agent = Agent(
    role="Weather Information Specialist",
    goal="Provide accurate weather information for specific locations and times",
    backstory="""You are a weather information specialist that must call the available tool to get the most recent reports""",
    verbose=False,
    allow_delegation=False,
    tools=[weather_tool],
    max_iter=5,
)

manager_agent = Agent(
    role="Information Manager",
    goal="Coordinate information requests and delegate to specialized agents",
    backstory="""You manage and coordinate information requests, delegating specialized
    queries to the appropriate experts. You ensure users get the most accurate and relevant
    information.""",
    verbose=False,
    allow_delegation=True,
    max_iter=10,
)

# Create tasks
weather_task = Task(
    description="""Find out the current weather at a specific location.""",
    expected_output="Complete weather report with temperature, rain and wave height information",
    agent=weather_agent,
)

manager_task = Task(
    description="""Process the user query about weather in Paris, France.
    Ensure the weather information is complete (with temperature, rain and wave height) and properly formatted.
    You must coordinate with the weather agent for this task.""",
    expected_output="Weather report for Paris",
    agent=manager_agent,
)

# Instantiate crew with a sequential process
crew = Crew(
    agents=[weather_agent],
    tasks=[manager_task, weather_task],
    verbose=False,
    manager_agent=manager_agent,
    process=Process.hierarchical,
)


@patronus.traced("weather-crew-ai")
def main():
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()
```
