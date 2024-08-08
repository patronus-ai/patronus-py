from _evaluators import simple_evaluator
from patronus import Client, simple_task

client = Client()


client.experiment(
    "MyExperiment",
    data=[
        {
            "evaluated_model_input": "John",
            "evaluated_model_gold_answer": "Hi John",
        },
        {
            "evaluated_model_input": "Alice",
            "evaluated_model_gold_answer": "Hello Alice",
        },
    ],
    task=simple_task(lambda input: f"Hi {input}"),
    evaluators=[simple_evaluator(lambda output, gold_answer: output == gold_answer)],
)
