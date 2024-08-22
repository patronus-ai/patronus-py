from patronus import Client, simple_task

cli = Client()

eval_patronus_is_similar = cli.remote_evaluator(
    evaluator="custom-small",
    profile_name="system:is-similar-to-gold-answer",
)

cli.experiment(
    "patronus-evals",
    data=[
        {"evaluated_model_input": "Foo", "evaluated_model_gold_answer": "Hi Foo"},
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
    evaluators=[eval_patronus_is_similar],
)
