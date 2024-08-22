from openai import OpenAI

from patronus import Client, evaluator, task, TaskResult

oai = OpenAI()
cli = Client()


@evaluator
def is_odd(evaluated_model_output: str) -> bool:
    return len([x for x in evaluated_model_output.split() if x.strip()]) % 2 == 1


@task
def call_llm(evaluated_model_system_prompt: str, evaluated_model_input: str) -> TaskResult:
    model_output = (
        oai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": evaluated_model_system_prompt},
                {"role": "user", "content": evaluated_model_input},
            ],
            temperature=0,
        )
        .choices[0]
        .message.content
    )
    return TaskResult(
        evaluated_model_output=model_output,
        evaluated_model_name="gpt-4o",
        evaluated_model_provider="openai",
        evaluated_model_params={"temperature": 0},
        evaluated_model_selected_model="gpt4-o",
    )


system_prompt = """\
Answer in the odd number of words (whitespace separated alphanumeric strings).
Samples:
INPUT:
what is 2 + 2?
OUTPUT: 4
---
INPUT:
Hello Adam
Output:
Hello Adam, how are you?
---
"""

cli.experiment(
    project_name="custom-task",
    data=[
        {
            "evaluated_model_system_prompt": system_prompt,
            "evaluated_model_input": "Foo",
        },
        {
            "evaluated_model_system_prompt": system_prompt,
            "evaluated_model_input": "Bar",
        },
        {
            "evaluated_model_system_prompt": system_prompt,
            "evaluated_model_input": "Bar",
        },
        {
            "evaluated_model_system_prompt": system_prompt,
            "evaluated_model_input": "Who as the first president of the US?",
        },
    ],
    task=call_llm,
    evaluators=[is_odd],
)
