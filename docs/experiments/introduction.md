# Introduction to Experiments

The Patronus Experimentation Framework provides a systematic way to evaluate, compare, and improve Large Language Model (LLM) applications.
By standardizing the evaluation process, the framework enables consistent testing across model versions, prompting strategies, and data inputs.

## What are Experiments?

In Patronus, an experiment is a structured evaluation that:

1. Processes a **dataset** of examples
2. Runs each example through a **task** function (optional)
3. Evaluates the output using one or more **evaluators**
4. Records and analyzes the results

This approach provides a comprehensive view of how your LLM application performs across different inputs,
making it easier to identify strengths, weaknesses, and areas for improvement.

## Key Concepts

### Dataset

A dataset in Patronus consists of examples that your models or systems will process.
Each example, represented as a `Row` object, can contain:

- Input data
- Context information
- Expected outputs (gold answers)
- Metadata
- And more...

Datasets can be loaded from various sources including JSON files, CSV files, Pandas DataFrames, or defined directly in your code.

### Task

A task is a function that processes each dataset example. Tasks typically:

- Receive a `Row` object from the dataset
- Perform some processing (like calling an LLM)
- Return a `TaskResult` containing the output

Tasks are optional - you can evaluate pre-existing outputs by including them directly in your dataset.

### Evaluators

Evaluators assess the quality of task outputs based on specific criteria. Patronus supports various types of evaluators:

- **Remote Evaluators**: Use Patronus's managed evaluation services
- **Custom Evaluators**: Your own evaluation logic.
    - **Function-based**: Simple functions decorated with @evaluator() that need to be wrapped with FuncEvaluatorAdapter when used in experiments.
    - **Class-based**: More powerful evaluators created by extending `StructuredEvaluator` (synchronous) or `AsyncStructuredEvaluator` (asynchronous) base classes with predefined interfaces.

Each evaluator produces an `EvaluationResult` containing scores, pass/fail status, explanations, and other metadata.

**Evaluator Weights**: You can assign weights to evaluators to indicate their relative importance in your evaluation strategy. Weights are stored as experiment metadata and can be provided as either strings or floats representing valid decimal numbers. See the [Using Evaluators](evaluators.md#evaluator-weights-experiments-only) page for detailed information.

### Chains

For more complex workflows, Patronus supports multi-stage evaluation chains where the output of one evaluation stage becomes the input for the next.
This allows for pipeline-based approaches to LLM evaluation.

## Why Use the Experimentation Framework?

The Patronus Experimentation Framework offers several advantages over ad-hoc evaluation approaches:

- **Consistency**: Standardized evaluation across models and time
- **Reproducibility**: Experiments can be re-run with the same configuration
- **Scalability**: Process large datasets efficiently with concurrent execution
- **Comprehensive Analysis**: Collect detailed metrics and explanations
- **Integration**: Built-in tracing and logging with the broader Patronus ecosystem

## Example: Basic Experiment

Here's a simple example of a Patronus experiment:

```python
# experiment.py

from patronus.evals import RemoteEvaluator
from patronus.experiments import run_experiment

# Define a simple task function
def my_task(row, **kwargs):
    return f"The answer is: {row.task_input}"

# Run the experiment
experiment = run_experiment(
    dataset=[
        {"task_input": "What is 2+2?", "gold_answer": "4"},
        {"task_input": "Who wrote Hamlet?", "gold_answer": "Shakespeare"}
    ],
    task=my_task,
    evaluators=[
        RemoteEvaluator("judge", "patronus:fuzzy-match")
    ]
)

experiment.to_csv("./experiment-result.csv")
```

You can run the experiment by simply executing the python file:

```shell
python ./exeriment.py
```

The output of the script should look similar to this:

```text
==================================
Experiment  Global/root-1742834029: 100%|██████████| 2/2 [00:04<00:00,  2.44s/sample]

patronus:fuzzy-match (judge) [link_idx=0]
-----------------------------------------
Count     : 2
Pass rate : 0
Mean      : 0.0
Min       : 0.0
25%       : 0.0
50%       : 0.0
75%       : 0.0
Max       : 0.0

Score distribution
Score Range          Count      Histogram
0.00 - 0.20          2          ####################
0.20 - 0.40          0
0.40 - 0.60          0
0.60 - 0.80          0
0.80 - 1.00          0
```

In the following sections, we'll explore how to set up, run, and analyze experiments in detail.
