from patronus import Client, simple_task, simple_evaluator

client = Client()

client.experiment(
    "MyExperiment",
    data=[
        {
            "evaluated_model_input": "Foo",
            "evaluated_model_gold_answer": "Hi Foo",
        },
        {
            "evaluated_model_input": "Bar",
            "evaluated_model_gold_answer": "Hello Bar",
        },
        {
            "evaluated_model_input": "Bar",
            "evaluated_model_gold_answer": "eloh Bar!",
        },
    ],
    task=simple_task(lambda input: f"Hi {input}"),
    evaluators=[simple_evaluator(lambda output, gold_answer: output == gold_answer)],
)
