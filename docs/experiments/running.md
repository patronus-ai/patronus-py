# Running Experiments

This page covers how to set up and run experiments using the Patronus Experimentation Framework.

## Basic Experiment Structure

A Patronus experiment requires at minimum:

- A dataset to process
- One or more evaluators to assess outputs

Additionally, most experiments will include:

- A task function that processes each dataset example
- Configuration options for tracing, logging, and concurrency

## Setting Up an Experiment

### The `run_experiment` Function

The main entry point for the framework is the `run_experiment()` function:

```python
from patronus.experiments import run_experiment

experiment = run_experiment(
    dataset=my_dataset,               # Required: What to evaluate
    task=my_task_function,            # Optional: How to process inputs
    evaluators=[my_evaluator],        # Required: How to assess outputs
    tags={"dataset-version": "v1.0"}, # Optional: Tags for the experiment
    max_concurrency=10,               # Optional: Control parallel execution
    project_name="My Project",        # Optional: Override the global project name
    experiment_name="Test Run"        # Optional: Name this experiment run
)
```

## Creating a Simple Experiment

Let's walk through a complete example:

```python
from patronus import evaluator, RemoteEvaluator
from patronus.experiments import run_experiment, FuncEvaluatorAdapter

dataset = [
    {
        "task_input": "What is the capital of France?",
        "gold_answer": "Paris"
    },
    {
        "task_input": "Who wrote Romeo and Juliet?",
        "gold_answer": "William Shakespeare"
    }
]

# Define a task (in a real scenario, this would call an LLM)
def answer_question(row, **kwargs):
    if "France" in row.task_input:
        return "The capital of France is Paris."
    elif "Romeo and Juliet" in row.task_input:
        return "Romeo and Juliet was written by William Shakespeare."
    return "I don't know the answer to that question."

@evaluator()
def contains_answer(task_result, row, **kwargs) -> bool:
    if not task_result or not row.gold_answer:
        return False
    return row.gold_answer.lower() in task_result.output.lower()

run_experiment(
    dataset=dataset,
    task=answer_question,
    evaluators=[
        # Use a Patronus-managed evaluator
        RemoteEvaluator("judge", "patronus:fuzzy-match"),

        # Use our custom evaluator
        FuncEvaluatorAdapter(contains_answer)
    ],
    tags={"model": "simulated", "version": "v1"}
)
```

## Experiment Execution Flow

When you call `run_experiment()`, the framework follows these steps:

1. **Preparation**: Initializes the experiment context and prepares the dataset
2. **Processing**: For each dataset row:
   - Runs the task function if provided
   - Passes the task output to the evaluators
   - Collects evaluation results
3. **Reporting**: Generates a summary of evaluation results
4. **Return**: Returns an `Experiment` object with the complete results

## Synchronous vs. Asynchronous Execution

The `run_experiment()` function detects whether it's being called from an async context:

- In a synchronous context, it will block until the experiment completes
- In an async context, it returns an awaitable that can be awaited

```python
# Synchronous usage:
experiment = run_experiment(dataset, task, evaluators)

# Asynchronous usage:
experiment = await run_experiment(dataset, task, evaluators)
```

## Manual Experiment Control

For more control over the experiment lifecycle, you can create and run an experiment manually:

```python
from patronus.experiments import Experiment

# Create the experiment
experiment = await Experiment.create(
    dataset=dataset,
    task=task,
    evaluators=evaluators,
    # Additional configuration options...
)

# Run the experiment when ready
experiment = await experiment.run()
```

This approach is useful when you need to perform additional setup between experiment creation and execution.

## Experiment Results

After an experiment completes, you can access the results in several ways:

```python
# Get a Pandas DataFrame
df = experiment.to_dataframe()

# Save to CSV
experiment.to_csv("results.csv")

# Access the built-in summary
# (This is automatically printed at the end of the experiment)
```

The experiment results include:

- Inputs from the dataset
- Task outputs
- Evaluation scores and pass/fail statuses
- Explanations and metadata
- Performance timing information

In the next sections, we'll explore datasets, tasks, and evaluators in more detail.
