import os

from patronus import init
from patronus import Client
from patronus.tracing import get_logger
from patronus.tracing.decorators import traced

"""
export PATRONUS_API_KEY='<your api key>'
"""

# Initialize Patronus
client = Client(
    # This is the default and can be omitted
    api_key=os.getenv("PATRONUS_API_KEY")
)
init(project_name="New Project")
logger = get_logger()


# Traced function example
@traced()
def evaluation_func(input: str, output: str, context: str):
    result = client.evaluate(
        evaluator="lynx",
        criteria="patronus:hallucination",
        evaluated_model_input=input,
        evaluated_model_output=output,
        evaluated_model_retrieved_context=context,
    )
    return result.pass_


@traced()
def demo_workflow(input: str, context: str):
    logger.debug("Starting my workflow.")
    evaluation_func(input=input, output="A dinosaur.", context=context)
    logger.debug("Workflow done.")


if __name__ == "__main__":
    demo_workflow(
        input="What is the biggest animal in the world",
        context="The biggest animal in the world is the blue whale (Balaenoptera musculus).",
    )
