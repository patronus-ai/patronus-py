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
uv run --no-cache --with "patronus-examples[langchain]" \
    -m patronus_examples.tracking.langchain_weather
```

### Running the script directly

If you've cloned the repository, you can run the script directly:

```shell
# Clone the repository
git clone https://github.com/patronus-ai/patronus-py.git
cd patronus-py

# Run the example script (requires uv)
./examples/patronus_examples/tracking/langchain_weather.py
```

### Manual installation

If you prefer to copy the example code to your own project, you'll need to install these dependencies:

```shell
pip install patronus
pip install pydantic
pip install langchain_openai
pip install langgraph
pip install langchain_core
pip install openinference-instrumentation-langchain
pip install opentelemetry-instrumentation-threading
pip install opentelemetry-instrumentation-asyncio
```

## Example overview

This example demonstrates how to use Patronus to trace a LangChain and LangGraph workflow for a weather application. The example:

1. Sets up a StateGraph with manager and weather agent nodes
2. Implements a router to control workflow transitions
3. Uses a tool to provide mock weather data
4. Traces the entire LangChain and LangGraph execution with Patronus

The example shows how Patronus can provide visibility into complex, multi-node LangGraph workflows, including tool usage and agent transitions.

## Example code

```python
# examples/patronus_examples/tracking/langchain_weather.py

from typing import Literal, Dict, List, Any
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor

import patronus

patronus.init(
    integrations=[
        LangChainInstrumentor(),
        ThreadingInstrumentor(),
        AsyncioInstrumentor(),
    ]
)


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a given city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        A string describing the current weather in the city.
    """
    return f"The weather in {city} is sunny"


class MessagesState(BaseModel):
    """State for the manager-weather agent workflow."""

    messages: List[BaseMessage] = Field(default_factory=list)
    current_agent: str = Field(default="manager")


manager_model = ChatOpenAI(temperature=0, model="gpt-4o")
weather_model = ChatOpenAI(temperature=0, model="gpt-4o")

tools = [get_weather]
tools_dict = {tool.name: tool for tool in tools}

weather_model_with_tools = weather_model.bind_tools(tools)


def manager_agent(state: MessagesState) -> Dict[str, Any]:
    messages = state.messages  # Access as attribute
    # Get response from the manager model
    response = manager_model.invoke(messages)

    # Check if the manager wants to use the weather agent
    manager_text = response.content.lower()
    if "weather" in manager_text and "in" in manager_text:
        # Delegate to weather agent
        return {
            "messages": messages
                        + [
                            AIMessage(
                                content="I'll check the weather for you. Delegating to weather agent."
                            )
                        ],
            "current_agent": "weather",
        }

    return {"messages": messages + [response], "current_agent": "manager"}


# Define the weather agent node using a simpler approach
def weather_agent(state: MessagesState) -> Dict[str, Any]:
    messages = state.messages  # Access as attribute
    human_queries = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not human_queries:
        return {
            "messages": messages + [AIMessage(content="I need a query about weather.")],
            "current_agent": "manager",
        }

    query = human_queries[-1].content

    try:
        # weather_prompt = (
        #     f"Extract the city name from this query and provide the weather: '{query}'"
        # )

        city_match = None

        # Common cities that might be mentioned
        common_cities = [
            "Paris",
            "London",
            "New York",
            "Tokyo",
            "Berlin",
            "Rome",
            "Madrid",
        ]
        for city in common_cities:
            if city.lower() in query.lower():
                city_match = city
                break

        if city_match:
            weather_result = get_weather.invoke(city_match)
            weather_response = (
                f"I checked the weather for {city_match}. {weather_result}"
            )
        else:
            if "weather in " in query.lower():
                parts = query.lower().split("weather in ")
                if len(parts) > 1:
                    city_match = parts[1].strip().split()[0].capitalize()
                    weather_result = get_weather.invoke(city_match)
                    weather_response = (
                        f"I checked the weather for {city_match}. {weather_result}"
                    )
                else:
                    weather_response = (
                        "I couldn't identify a specific city in your query."
                    )
            else:
                weather_response = "I couldn't identify a specific city in your query."

        return {
            "messages": messages
                        + [AIMessage(content=f"Weather Agent: {weather_response}")],
            "current_agent": "manager",
        }
    except Exception as e:
        error_message = f"I encountered an error while checking the weather: {str(e)}"
        return {
            "messages": messages
                        + [AIMessage(content=f"Weather Agent: {error_message}")],
            "current_agent": "manager",
        }


def router(state: MessagesState) -> Literal["manager", "weather", END]:
    if len(state.messages) > 10:  # Prevent infinite loops
        return END

    # Route based on current_agent
    if state.current_agent == "weather":
        return "weather"
    elif state.current_agent == "manager":
        # Check if the last message is from the manager and indicates completion
        if len(state.messages) > 0 and isinstance(state.messages[-1], AIMessage):
            if "delegating to weather agent" not in state.messages[-1].content.lower():
                return END

    return "manager"


workflow = StateGraph(MessagesState)
workflow.add_node("manager", manager_agent)
workflow.add_node("weather", weather_agent)

workflow.set_entry_point("manager")
workflow.add_conditional_edges("manager", router)
workflow.add_conditional_edges("weather", router)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


def run_workflow(query: str):
    initial_state = MessagesState(
        messages=[HumanMessage(content=query)], current_agent="manager"
    )

    config = {"configurable": {"thread_id": "weather_demo_thread"}}
    final_state = app.invoke(initial_state, config=config)

    for message in final_state["messages"]:
        if isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}")
        else:
            print(f"Other: {message.content}")

    return final_state


@patronus.traced("weather-langchain")
def main():
    final_state = run_workflow("What is the weather in Paris?")
    return final_state


if __name__ == "__main__":
    main()
```
