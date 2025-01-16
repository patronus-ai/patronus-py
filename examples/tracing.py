import os
import time

from patronus import Client
from patronus.tracing.trace import traced

"""
export PATRONUS_API_KEY='<your api key>'
"""

# Initialize Patronus Client
client = Client(api_key=os.getenv('PATRONUS_API_KEY'))

###### CASE 1 #######
# Simple tracing example
#
#
# Expected output:
# Span(name="this_is_my_func")
# -- Log(input.kwargs={"sleep_time": 3}, output="I've been sleeping for 3s.")

@traced()
def this_is_my_func(sleep_time: int, *args, **kwargs):
    print("Doing magic...")
    time.sleep(sleep_time)
    print("Done...")
    return f"I've been sleeping for {time}s."


###### CASE 2 #######
# Simple code tracing nested
#
#
# Expected output:
# Span(name="root_func")
# Log(output="DONE")
# -- Span(name="this_is_my_func")
# ---- Log(input.args=1, output='I've been sleeping for 1s.')
# -- Span(name="this_is_my_func")
# ---- Log(input.args=2, output='I've been sleeping for 3s.')
# -- Span(name="this_is_my_func")
# ---- Log(input.args=3, output='I've been sleeping for 3s.')
def not_traced_func(*args, **kwargs):
    return "Nothing"

@traced()
def root_func():
    this_is_my_func(1)
    this_is_my_func(2)
    this_is_my_func(3)
    not_traced_func()
    return "DONE"



###### CASE 3 #######
### Trace with evaluation
#
#
# Expected output:
# Span(name="my_evaluation")
# -- Log(output: 1)
# -- Log(type='patronus-evaluation', evaluated_model_input="What is the largest animal in the world?", evaluated_model_output="The giant sandworm.", evaluated_model_retrieved_context="The blue whale is the largest known animal.",)

@traced()
def my_evaluation():
    result = client.evaluate(
        evaluator="lynx-small",
        criteria="patronus:hallucination",
        evaluated_model_input="What is the largest animal in the world?",
        evaluated_model_output="The giant sandworm.",
        evaluated_model_retrieved_context="The blue whale is the largest known animal.",
        tags={"scenario": "onboarding"},
    )
    return result.pass_


###### CASE 4 #######
### Nested with evaluation
#
#
# Expected output:
# Span(name="my_workflow")
# -- Log("input.args": ["WorkflowDemo1"], "output": "Done: 2"}
# -- Span(name="client_evaluate_lynx"}
# ---- Log(
#         "input.args": ["What is the largest animal in the world?", "The giant sandworm.", "The blue whale is the largest known animal."]
#         "output": 0
#     )
# ---- Log(type='patronus-evaluation', evaluated_model_input="What is the largest animal in the world?", evaluated_model_output="The giant sandworm.", evaluated_model_retrieved_context="The blue whale is the largest known animal.",)
#
# -- Span(name="client_evaluate_lynx"}
# ---- Log(
#         "input.args": ["What is the smallest animal in the world?", "A rat.", ""]
#         "output": 0
#     )
# ---- Log(type='patronus-evaluation', evaluated_model_input="What is the smallest animal in the world?", evaluated_model_output="A rat", evaluated_model_retrieved_context="",)

@traced()
def client_evaluate_lynx(input, output, context):
    result = client.evaluate(
        evaluator="lynx-small",
        criteria="patronus:hallucination",
        evaluated_model_input=input,
        evaluated_model_output=output,
        evaluated_model_retrieved_context=context,
        tags={"scenario": "onboarding"},
    )
    return result.pass_

@traced()
def my_workflow(workflow_subject: str):
    data = [
        ("What is the largest animal in the world?", "The giant sandworm.", "The blue whale is the largest known animal."),
        ("What is the smallest animal in the world?", "A rat.", "")
    ]
    for d in data:
        client_evaluate_lynx(d[0], d[1], d[2])
    return f"Done: {len(data)}"


# TODO: Remove after testing:
def case(n: int):
    case_func_map = {
        1: lambda: this_is_my_func(3),
        2: lambda: root_func(),
        3: lambda: my_evaluation(),
        4: lambda: my_workflow("WorkflowDemo1"),
    }
    case_func_map[n]()

case(1)



@traced()
def this_is_my_func(sleep_time: int, *args, **kwargs):
    """
    Span(Group Evaluations)
    --- Span (local eval)
    ------ Log (evaluation log)
    ------ Evaluation()
    --- Span (extr eval)
    ------- Log (evaluation log)
    ------ Evaluation()
    """
