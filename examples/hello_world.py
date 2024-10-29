import os
from patronus import Client, Row, TaskResult, evaluator, task

client = Client(
    # This is the default and can be omitted
    api_key=os.environ.get("PATRONUS_API_KEY"),
)


@task
def my_task(row: Row):
    return f"{row.evaluated_model_input} World"


@evaluator
def exact_match(row: Row, task_result: TaskResult):
    return task_result.evaluated_model_output == row.evaluated_model_gold_answer


client.experiment(
    "Tutorial Project",
    dataset=[
        {
            "evaluated_model_input": "Hello",
            "evaluated_model_gold_answer": "Hello World",
        },
    ],
    task=my_task,
    evaluators=[exact_match],
)
