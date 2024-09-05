import textwrap

from patronus import Client

cli = Client()

evaluate_proper_language = cli.remote_evaluator(
    "custom-large",
    "detect-requested-programming-languages",
    profile_config={
        "pass_criteria": textwrap.dedent(
            """
            The MODEL OUTPUT should provide only valid code in a well-known programming language.
            The MODEL OUTPUT should consist of the code in a programming language specified in the USER INPUT.
            """
        ),
    },
    allow_update=True,
)

data = [
    {
        "evaluated_model_input": "Write a hello world example in Python.",
        "evaluated_model_output": "print('Hello World!')",
    },
    {
        "evaluated_model_input": "Write a hello world example in JavaScript.",
        "evaluated_model_output": "print('Hello World!')",
    },
]

cli.experiment(
    "Programming Language Detection",
    data=data,
    evaluators=[evaluate_proper_language],
)
