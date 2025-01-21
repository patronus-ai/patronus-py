import os

from patronus import Client
from patronus.tracing import init, get_logger
from patronus.tracing.decorators import traced

"""
export PATRONUS_API_KEY='<your api key>'
"""

# Initialize Patronus
client = Client(api_key=os.getenv('PATRONUS_API_KEY'))
init(project_name="Global")
logger = get_logger("Global")


# Traced function example
@traced()
def evaluation_func(input: str, output: str, context: str):
    result = client.evaluate(
        evaluator="hallucination",
        criteria="patronus:hallucination",
        evaluated_model_input=input,
        evaluated_model_output=output,
        evaluated_model_retrieved_context=context,
    )
    return result.pass_


@traced()
def demo_workflow(input: str, context: str):
    logger.debug("Starting my workflow.")
    evaluation_func(input=input, output='A dinosaur.', context=context)
    logger.debug("Workflow done.")

demo_workflow(input="What is the biggest animal in the world", context="The biggest animal in the world is the blue whale (Balaenoptera musculus).")
