# Working with Datasets

Datasets provide the foundation for Patronus experiments, containing the examples that your tasks and evaluators will process.
This page explains how to create, load, and work with datasets effectively.

## Dataset Structure and Evaluator Compatibility

Patronus experiments are designed to work with `StructuredEvaluator` classes, which expect specific input parameters.
The standard dataset fields map directly to these parameters, making integration seamless:

- `system_prompt`: System instruction for LLM-based tasks
- `task_context`: Additional information or context (string or list of strings)
- `task_metadata`: Additional structured information about the task
- `task_attachments`: Files or other binary data
- `task_input`: The primary input query or text
- `task_output`: The model's response or output to evaluate
- `gold_answer`: The expected correct answer or reference output
- `tags`: Key-value pairs
- `sid`: A unique identifier for the example (automatically generated if not provided)

While you can include any custom fields in your dataset, using these standard field names ensures compatibility with structured evaluators without additional configuration.

## Creating Datasets

Patronus accepts datasets in several formats:

### List of Dictionaries

```python
dataset = [
    {
        "task_input": "What is machine learning?",
        "gold_answer": "Machine learning is a subfield of artificial intelligence...",
        "tags": {"category": "ai", "difficulty": "beginner"},
        "difficulty": "beginner"  # Custom field
    },
    {
        "task_input": "Explain quantum computing",
        "gold_answer": "Quantum computing uses quantum phenomena...",
        "tags": {"category": "physics", "difficulty": "advanced"},
        "difficulty": "advanced"  # Custom field
    }
]

experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[my_evaluator]
)
```

### Pandas DataFrame

```python
import pandas as pd

df = pd.DataFrame({
    "task_input": ["What is Python?", "What is JavaScript?"],
    "gold_answer": ["Python is a programming language...", "JavaScript is a programming language..."],
    "tags": [{"type": "backend"}, {"type": "frontend"}],
    "language_type": ["backend", "frontend"]  # Custom field
})

experiment = run_experiment(dataset=df, ...)
```

### CSV or JSONL Files

```python
from patronus.datasets import read_csv, read_jsonl

# Load with default field mappings
dataset = read_csv("questions.csv")

# Load with custom field mappings
dataset = read_jsonl(
    "custom.jsonl",
    task_input_field="question",     # Map "question" field to "task_input"
    gold_answer_field="answer",      # Map "answer" field to "gold_answer"
    system_prompt_field="instruction", # Map "instruction" field to "system_prompt"
    tags_field="metadata"            # Map "metadata" field to "tags"
)
```

### Remote Datasets

Patronus allows you to work with datasets stored remotely on the Patronus platform.
This is useful for sharing standard datasets across your organization or utilizing pre-built evaluation datasets.

```python
from patronus.datasets import RemoteDatasetLoader

# Load a dataset from the Patronus platform using its name
remote_dataset = RemoteDatasetLoader("financebench")

# Load a dataset from the Patronus platform using its ID
remote_dataset = RemoteDatasetLoader(by_id="d-eo6a5zy3nwach69b")

experiment = run_experiment(
    dataset=remote_dataset,
    task=my_task,
    evaluators=[my_evaluator],
)
```

The `RemoteDatasetLoader` asynchronously fetches the dataset from the Patronus API when the experiment runs.
It handles the data mapping automatically, transforming the API response into the standard dataset structure
with all the expected fields (`system_prompt`, `task_input`, `gold_answer`, etc.).

Remote datasets follow the same structure and field conventions as local datasets, making them interchangeable in your experiment code.

## Accessing Dataset Fields

During experiment execution, dataset examples are provided as `Row` objects:

```python
def my_task(row, **kwargs):
    # Access standard fields
    question = row.task_input
    reference = row.gold_answer
    context = row.task_context

    # Access tags
    if row.tags:
        category = row.tags.get("category")

    # Access custom fields directly
    difficulty = row.difficulty  # Access custom field by name

    # Access row ID
    sample_id = row.sid

    return f"Answering {difficulty} question (ID: {sample_id}): {question}"
```

The `Row` object automatically provides attributes for all fields in your dataset, making access straightforward for both standard and custom fields.

## Using Custom Dataset Schemas

If your dataset uses a different schema than the standard field names, you have two options:

1. **Map fields during loading**: Use field mapping parameters when loading data
   ```python
   from patronus.datasets import read_csv

   dataset = read_csv("data.csv",
                     task_input_field="question",
                     gold_answer_field="answer",
                     tags_field="metadata")
   ```

2. **Use evaluator adapters**: Create adapters that transform your data structure to match what evaluators expect

   ```python
   from patronus import evaluator
   from patronus.experiments import run_experiment, FuncEvaluatorAdapter

   @evaluator()
   def my_evaluator_function(*, expected, actual, context):
       ...

   class CustomAdapter(FuncEvaluatorAdapter):
       def transform(self, row, task_result, parent, **kwargs):
           # Transform dataset fields to evaluator parameters.
           # The first value is list of positional arguments (*args) passed to the evaluator function.
           # The second value is named arguments (**kwargs) passed to the evaluator function.
           return [], {
               "expected": row.reference_answer,  # Map custom field to expected parameter
               "actual": task_result.output if task_result else None,
               "context": row.additional_info    # Map custom field to context parameter
           }

   experiment = run_experiment(
       dataset=custom_dataset,
       evaluators=[CustomAdapter(my_evaluator_function)]
   )
   ```

This adapter approach is particularly important for function-based evaluators, which need to be explicitly adapted for use in experiments.

## Dataset IDs and Sample IDs

Each dataset and row can have identifiers that are used for organization and tracing:

```python
from patronus.datasets import Dataset

# Dataset with explicit ID
dataset = Dataset.from_records(
    records=[...],
    dataset_id="qa-dataset-v1"
)

# Dataset with explicit sample IDs
dataset = Dataset.from_records([
    {"sid": "q1", "task_input": "Question 1", "gold_answer": "Answer 1"},
    {"sid": "q2", "task_input": "Question 2", "gold_answer": "Answer 2"}
])
```

If not provided, sample IDs (`sid`) are automatically generated.

## Best Practices

1. **Use standard field names when possible**: This minimizes the need for custom adapters
2. **Include gold answers**: This enables more comprehensive evaluation
3. **Use tags for organization**: Tags provide a flexible way to categorize examples
4. **Keep task inputs focused**: Clear, concise inputs lead to better evaluations
5. **Add relevant metadata**: Additional context helps with result analysis
6. **Normalize data before experiments**: Pre-process data to ensure consistent format
7. **Consider remote datasets for team collaboration**: Use the Patronus platform to share standardized datasets

In the next section, we'll explore how to create tasks that process your dataset examples.
