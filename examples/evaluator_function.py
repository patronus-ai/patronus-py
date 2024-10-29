from patronus import Client, evaluator, Row

client = Client()


@evaluator
def iexact_match(row: Row) -> bool:
    return row.evaluated_model_output.lower().strip() == row.evaluated_model_gold_answer.lower().strip()


client.experiment(
    "Tutorial",
    dataset=[
        {
            "evaluated_model_input": "Translate 'Good night' to French.",
            "evaluated_model_output": "bonne nuit",
            "evaluated_model_gold_answer": "Bonne nuit",
        },
        {
            "evaluated_model_input": "Summarize: 'AI improves efficiency'.",
            "evaluated_model_output": "ai improves efficiency",
            "evaluated_model_gold_answer": "AI improves efficiency",
        },
    ],
    evaluators=[iexact_match],
    experiment_name="Case Insensitive Match",
)
