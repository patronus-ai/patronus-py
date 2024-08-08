from patronus import Client, evaluator, simple_task

cli = Client()


@evaluator
def iexact_match(evaluated_model_output: str, evaluated_model_gold_answer: str) -> bool:
    return evaluated_model_output.lower().strip() == evaluated_model_gold_answer.lower().strip()


eval_patronus_is_similar = cli.remote_evaluator(
    family="custom",
    evaluator="custom-small",
    profile_name="system:is-similar-to-gold-answer",
)

cli.experiment(
    "mixed",
    data=[
        {"evaluated_model_input": "Foo", "evaluated_model_gold_answer": "hi foo"},
        {
            "evaluated_model_input": "Bar",
            "evaluated_model_gold_answer": "Hello bar",
        },
        {
            "evaluated_model_input": "Bar",
            "evaluated_model_gold_answer": "eloh Bar!",
        },
    ],
    task=simple_task(lambda input: f"Hi {input}"),
    evaluators=[iexact_match, eval_patronus_is_similar],
)
