import os
from patronus import Client, simple_task, simple_evaluator

client = Client(
    # This is the default and can be omitted
    api_key=os.environ.get("PATRONUS_API_KEY"),
)

task = simple_task(lambda input: f"{input} World")

exact_match = simple_evaluator(lambda output, gold_answer: output == gold_answer)

client.experiment(
    "Hello World",
    data=[
        {
            "evaluated_model_input": "Hello",
            "evaluated_model_gold_answer": "Hello World",
        },
    ],
    task=task,
    evaluators=[exact_match],
)
