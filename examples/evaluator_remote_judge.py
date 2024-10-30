import textwrap

from patronus import Client


client = Client()

evaluate_proper_language = client.remote_evaluator(
    "judge-large",
    "detect-requested-programming-languages",
    criteria_config={
        "pass_criteria": textwrap.dedent(
            """
            The MODEL OUTPUT should provide only valid code in any well-known programming language.
            The MODEL OUTPUT should consist of the code in a programming language specified in the USER INPUT.
            """
        ),
    },
    allow_update=True,
)

dataset = [
    {
        "evaluated_model_input": "Write a hello world example in Python.",
        "evaluated_model_output": "print('Hello World!')",
    },
    {
        "evaluated_model_input": "Write a hello world example in JavaScript.",
        "evaluated_model_output": "print('Hello World!')",
    },
]

client.experiment(
    "Tutorial",
    dataset=dataset,
    evaluators=[evaluate_proper_language],
    experiment_name="Detect Programming Languages",
)
