# Advanced Experiment Features

This page covers advanced features of the Patronus Experimentation Framework that help you build more sophisticated evaluation workflows.

## Multi-Stage Processing with Chains

For complex workflows, you can use chains to create multi-stage processing and evaluation pipelines.
Chains connect multiple processing stages where the output of one stage becomes the input to the next.

### Basic Chain Structure

```python
from patronus.experiments import run_experiment
from patronus.evals import RemoteEvaluator

experiment = run_experiment(
    dataset=dataset,
    chain=[
        # Stage 1: Generate summaries
        {
            "task": generate_summary,
            "evaluators": [
                RemoteEvaluator("judge", "conciseness"),
                RemoteEvaluator("judge", "coherence")
            ]
        },
        # Stage 2: Generate questions from summaries
        {
            "task": generate_questions,
            "evaluators": [
                RemoteEvaluator("judge", "relevance"),
                QuestionDiversityEvaluator()
            ]
        },
        # Stage 3: Answer questions
        {
            "task": answer_questions,
            "evaluators": [
                RemoteEvaluator("judge", "factual-accuracy"),
                RemoteEvaluator("judge", "helpfulness")
            ]
        }
    ]
)
```

Each stage in the chain can:
1. Apply its own task function (or no task if set to `None`)
2. Use its own set of evaluators
3. Access results from previous stages

### Accessing Previous Results in Chain Tasks

Tasks in later chain stages can access outputs and evaluations from earlier stages through the `parent` parameter:

```python
def generate_questions(row, parent, **kwargs):
    """Generate questions based on a summary from the previous stage."""
    # Get the summary from the previous task
    summary = parent.task.output if parent and parent.task else None

    if not summary:
        return None

    # Check if summary evaluations are available
    if parent and parent.evals:
        coherence = parent.evals.get("judge:coherence")
        # Use previous evaluation results to guide question generation
        if coherence and coherence.score > 0.8:
            return "Here are three detailed questions based on the summary..."
        else:
            return "Here are three basic questions about the summary..."

    # Default questions if no evaluations available
    return "Here are some standard questions about the topic..."
```

This example demonstrates how a task can adapt its behavior based on previous outputs and evaluations.

## Concurrency Controls

For better performance, the framework automatically processes dataset examples concurrently.
You can control this behavior to prevent rate limiting or resource exhaustion:

```python
experiment = run_experiment(
    dataset=large_dataset,
    task=api_intensive_task,
    evaluators=[evaluator1, evaluator2],
    # Limit the number of concurrent tasks and evaluations
    max_concurrency=5
)
```

This is particularly important for:
- Tasks that make API calls with rate limits
- Resource-intensive processing
- Large datasets with many examples

## OpenTelemetry Integrations

The framework supports OpenTelemetry instrumentation for enhanced tracing and monitoring:

```python
from openinference.instrumentation.openai import OpenAIInstrumentor

experiment = run_experiment(
    dataset=dataset,
    task=openai_task,
    evaluators=[evaluator1, evaluator2],
    # Add OpenTelemetry instrumentors
    integrations=[OpenAIInstrumentor()]
)
```

Benefits of OpenTelemetry integration include:
- Automatic capture of API calls and parameters
- Detailed timing information for performance analysis
- Integration with observability platforms

## Organizing Experiments

### Custom Experiment Names and Projects

Organize your experiments into projects with descriptive names for better management:

```python
experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[evaluator1, evaluator2],
    # Organize experiments
    project_name="RAG System Evaluation",
    experiment_name="baseline-gpt4-retrieval"
)
```

The framework automatically appends a timestamp to experiment names for uniqueness.

### Tags for Filtering and Organization

Tags help organize and filter experiment results:

```python
experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[evaluator1, evaluator2],
    # Add tags for filtering and organization
    tags={
        "model": "gpt-4",
        "version": "2.0",
        "retrieval_method": "bm25",
        "environment": "staging"
    }
)
```

Important notes about tags:

- Tags are propagated to all evaluation results in the experiment
- They cannot be overridden by tasks or evaluators
- Use a small set of consistent values for each tag (avoid having too many unique values)
- Tags are powerful for filtering and grouping in analysis

### Experiment Metadata

Experiments automatically capture important metadata, including evaluator weights when specified:

```python
from patronus.experiments import run_experiment, FuncEvaluatorAdapter
from patronus.evals import RemoteEvaluator
from patronus import evaluator

@evaluator()
def custom_check(row, **kwargs):
    return True

# Experiment with weighted evaluators
experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[
        RemoteEvaluator("judge", "patronus:is-concise", weight=0.6),
        FuncEvaluatorAdapter(custom_check, weight="0.4")
    ]
)

# Weights are automatically stored in experiment metadata
# as "evaluator_weights": {
#     "judge:patronus:is-concise": "0.6",
#     "custom_check:": "0.4"
# }
```

Evaluator weights are automatically collected and stored in the experiment's metadata under the `evaluator_weights` key. This provides a permanent record of how evaluators were weighted in each experiment for reproducibility and analysis.

For more details on using evaluator weights, see the [Using Evaluators](evaluators.md#evaluator-weights-experiments-only) page.

## Custom API Configuration

For on-prem environments, you can customize the API configuration:

```python
experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[evaluator1, evaluator2],
    # Custom API configuration
    api_key="your-api-key",
    api_url="https://custom-endpoint.patronus.ai",
    otel_endpoint="https://custom-telemetry.patronus.ai",
    timeout_s=120
)
```

## Manual Experiment Control

For fine-grained control over the experiment lifecycle, you can create and run experiments manually:

```python
from patronus.experiments import Experiment

# Create the experiment
experiment = await Experiment.create(
    dataset=dataset,
    task=task,
    evaluators=evaluators,
    # Additional configuration...
)

# Perform custom setup if needed
# ...

# Run the experiment when ready
await experiment.run()

# Export results
experiment.to_csv("results.csv")
```

This pattern is useful when you need to:
- Perform additional setup after experiment creation
- Control exactly when execution starts
- Implement custom pre- or post-processing

## Best Practices

When using advanced experiment features:

1. **Start simple**: Begin with basic experiments before adding chain complexity
2. **Test incrementally**: Validate each stage before combining them
3. **Monitor resources**: Watch for memory usage with large datasets
4. **Set appropriate concurrency**: Balance throughput against rate limits
5. **Use consistent tags**: Create a standard tagging system across experiments
