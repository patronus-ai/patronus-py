from patronus import evaluator, simple_task, Client

cli = Client()


@evaluator
def iexact_match(evaluated_model_output: str, evaluated_model_gold_answer: str) -> bool:
    return evaluated_model_output.lower().strip() == evaluated_model_gold_answer.lower().strip()


cli.experiment(
    "local-imatch-evals",
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
    evaluators=[iexact_match],
)
