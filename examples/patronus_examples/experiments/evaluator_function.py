from patronus import evaluator
from patronus.experiments import run_experiment, Row, FuncEvaluatorAdapter


@evaluator
def iexact_match(row: Row, **kwargs) -> bool:
    return row.task_output.lower().strip() == row.gold_answer.lower().strip()


run_experiment(
    project_name="Tutorial",
    dataset=[
        {
            "task_input": "Translate 'Good night' to French.",
            "task_output": "bonne nuit",
            "gold_answer": "Bonne nuit",
        },
        {
            "task_input": "Summarize: 'AI improves efficiency'.",
            "task_output": "ai improves efficiency",
            "gold_answer": "AI improves efficiency",
        },
    ],
    evaluators=[FuncEvaluatorAdapter(iexact_match)],
    experiment_name="Case Insensitive Match",
)
