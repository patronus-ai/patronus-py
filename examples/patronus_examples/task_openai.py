from openai import OpenAI
import textwrap
from patronus import Client, task, TaskResult

oai = OpenAI()
cli = Client()


@task
def call_gpt(
    evaluated_model_system_prompt: str, evaluated_model_input: str
) -> TaskResult:
    model = "gpt-4o"
    params = {
        "temperature": 1,
        "max_tokens": 200,
    }
    evaluated_model_output = (
        oai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": evaluated_model_system_prompt},
                {"role": "user", "content": evaluated_model_input},
            ],
            **params,
        )
        .choices[0]
        .message.content
    )
    return TaskResult(
        evaluated_model_output=evaluated_model_output,
        metadata={
            "evaluated_model_name": model,
            "evaluated_model_provider": "openai",
            "evaluated_model_params": params,
            "evaluated_model_selected_model": model,
        },
        tags={"task_type": "chat_completion", "language": "English"},
    )


evaluate_on_point = cli.remote_evaluator(
    "custom-large",
    "is-on-point",
    criteria_config={
        "pass_criteria": textwrap.dedent(
            """
            The MODEL OUTPUT should accurately and concisely answer the USER INPUT.
            """
        ),
    },
    allow_update=True,
)

data = [
    {
        "evaluated_model_system_prompt": "You are a helpful assistant.",
        "evaluated_model_input": "How do I write a Python function?",
    },
    {
        "evaluated_model_system_prompt": "You are a knowledgeable assistant.",
        "evaluated_model_input": "Explain the concept of polymorphism in OOP.",
    },
    {
        "evaluated_model_system_prompt": "You are a creative poet who loves abstract ideas.",
        "evaluated_model_input": "What is 2 + 2?",
    },
]

cli.experiment(
    "Tutorial",
    data=data,
    task=call_gpt,
    evaluators=[evaluate_on_point],
    tags={"unit": "R&D", "version": "0.0.1"},
    experiment_name="OpenAI Task",
)
