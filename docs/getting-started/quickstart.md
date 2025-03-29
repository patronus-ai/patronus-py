# Quickstart

This guide will help you get started with the Patronus SDK through three practical examples.
We'll explore tracing, evaluation, and experimentation to give you a hands-on introduction to the core features.

## Initialization

Before running any of the examples, initialize the Patronus SDK:

```python
import os
import patronus

# Initialize with your API key
patronus.init(
    # This is the default and can be omitted
    api_key=os.environ.get("PATRONUS_API_KEY")
)
```

You can also use a configuration file instead of direct initialization:

```yaml
# patronus.yaml

api_key: "your-api-key"
project_name:  "Global"
app: "default"
```

For experiments, you don't need to explicitly call [`init()`][patronus.init.init] as [`run_experiment()`][patronus.experiments.run_experiment] handles initialization automatically.

## Example 1: Tracing with a Functional Evaluator

This example demonstrates how to trace function execution and create a simple functional evaluator.

```python
import patronus
from patronus import evaluator, traced

patronus.init()

@evaluator()
def exact_match(expected: str, actual: str) -> bool:
    return expected.strip() == actual.strip()

@traced()
def process_query(query: str) -> str:
    # In a real application, this would call an LLM
    return f"Processed response for: {query}"

# Use the traced function and evaluator together
@traced()
def main():
    query = "What is machine learning?"
    response = process_query(query)
    print(f"Response: {response}")

    expected_response = "Processed response for: What is machine learning?"
    result = exact_match(expected_response, response)
    print(f"Evaluation result: {result}")

if __name__ == "__main__":
    main()
```

In this example:

1. We created a simple `exact_match` evaluator using the `@evaluator()` decorator
2. We traced the `process_query` function using the `@traced()` decorator
3. We ran an evaluation by calling the evaluator function directly

The tracing will automatically capture execution details, timing, and results, making them available in the Patronus platform.

## Example 2: Using a Patronus Evaluator

This example shows how to use a Patronus Evaluator to assess model outputs for hallucinations.

```python
import patronus
from patronus import traced
from patronus.evals import RemoteEvaluator

patronus.init()


@traced()
def generate_insurance_response(query: str) -> str:
    # In a real application, this would call an LLM
    return "To even qualify for our car insurance policy, you need to have a valid driver's license that expires later than 2028."


@traced("Quickstart: detect hallucination")
def main():
    check_hallucinates = RemoteEvaluator("lynx", "patronus:hallucination")

    context = """
    To qualify for our car insurance policy, you need a way to show competence
    in driving which can be accomplished through a valid driver's license.
    You must have multiple years of experience and cannot be graduating from driving school before or on 2028.
    """

    query = "What is the car insurance policy?"
    response = generate_insurance_response(query)
    print(f"Query: {query}")
    print(f"Response: {response}")

    # Evaluate the response for hallucinations
    resp = check_hallucinates.evaluate(
        task_input=query,
        task_context=context,
        task_output=response
    )

    # Print the evaluation results
    print(f"""
Hallucination evaluation:
Passed: {resp.pass_}
Score: {resp.score}
Explanation: {resp.explanation}
""")

if __name__ == "__main__":
    main()
```

In this example:

1. We created a traced function generate_insurance_response to simulate an LLM response
2. We used the Patronus Lynx Evaluator
3. We evaluated whether the response contains information not supported by the context
4. We displayed the detailed evaluation results

Patronus Evaluators run on Patronus infrastructure and provide sophisticated assessment capabilities without requiring you to implement complex evaluation logic.

## Example 3: Running an Experiment with OpenAI

This example demonstrates how to run a comprehensive experiment to evaluate OpenAI model performance across multiple samples and criteria.

Before running Example 3, you'll need to install Pandas and the OpenAI SDK and OpenInference instrumentation:

```shell
pip install pandas openai openinference-instrumentation-openai
```

The OpenInference instrumentation automatically adds spans for all OpenAI API calls, capturing prompts, responses,
and model parameters without any code changes.
These details will appear in your Patronus traces for complete visibility into model interactions.

```python
from typing import Optional
import os

import patronus
from patronus.evals import evaluator, RemoteEvaluator, EvaluationResult
from patronus.experiments import run_experiment, FuncEvaluatorAdapter, Row, TaskResult
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor

oai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

patronus.init()


@evaluator()
def fuzzy_match(row: Row, task_result: TaskResult, **kwargs) -> Optional[EvaluationResult]:
    if not row.gold_answer or not task_result:
        return None

    gold_answer = row.gold_answer.lower()
    response = task_result.output.lower()

    key_terms = [term.strip() for term in gold_answer.split(',')]
    matches = sum(1 for term in key_terms if term in response)
    match_ratio = matches / len(key_terms) if key_terms else 0

    # Return a score between 0-1 indicating match quality
    return EvaluationResult(
        pass_=match_ratio > 0.7,
        score=match_ratio,
    )


def rag_task(row, **kwargs):
    # In a real RAG system, this would retrieve context before calling the LLM
    prompt = f"""
    Based on the following context, answer the question.

    Context:
    {row.task_context}

    Question: {row.task_input}

    Answer:
    """

    # Call OpenAI to generate a response
    response = oai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that answers questions based only on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )

    return response.choices[0].message.content


test_data = [
    {
        "task_input": "What is the main impact of climate change on coral reefs?",
        "task_context": """
        Climate change affects coral reefs through several mechanisms. Rising sea temperatures can cause coral bleaching,
        where corals expel their symbiotic algae and turn white, often leading to death. Ocean acidification, caused by
        increased CO2 absorption, makes it harder for corals to build their calcium carbonate structures. Sea level rise
        can reduce light availability for photosynthesis. More frequent and intense storms damage reef structures. The
        combination of these stressors is devastating to coral reef ecosystems worldwide.
        """,
        "gold_answer": "coral bleaching, ocean acidification, reduced calcification, habitat destruction"
    },
    {
        "task_input": "How do quantum computers differ from classical computers?",
        "task_context": """
        Classical computers process information in bits (0s and 1s), while quantum computers use quantum bits or qubits.
        Qubits can exist in multiple states simultaneously thanks to superposition, allowing quantum computers to process
        vast amounts of information in parallel. Quantum entanglement enables qubits to be correlated in ways impossible
        for classical bits. While classical computers excel at everyday tasks, quantum computers potentially have advantages
        for specific problems like cryptography, simulation of quantum systems, and certain optimization tasks. However,
        quantum computers face significant challenges including qubit stability, error correction, and scaling up to useful sizes.
        """,
        "gold_answer": "qubits instead of bits, superposition, entanglement, parallel processing"
    }
]

evaluators = [
    FuncEvaluatorAdapter(fuzzy_match),
    RemoteEvaluator("answer-relevance", "patronus:answer-relevance")
]

# Run the experiment with OpenInference instrumentation
print("Running RAG evaluation experiment...")
experiment = run_experiment(
    dataset=test_data,
    task=rag_task,
    evaluators=evaluators,
    tags={"system": "rag-prototype", "model": "gpt-3.5-turbo"},
    integrations=[OpenAIInstrumentor()]
)

# Export results to CSV (optional)
# experiment.to_csv("rag_evaluation_results.csv")
```

In this example:

1. We defined a task function `answer_questions` that generates responses for our experiment
2. We created a custom evaluator `contains_key_information` to check for specific content
3. We set up an experiment with multiple evaluators (both remote and custom)
4. We ran the experiment across a dataset of questions

Experiments provide a powerful way to systematically evaluate your LLM applications across multiple samples and criteria, helping you identify strengths and weaknesses in your models.
