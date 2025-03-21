# User-Defined Evaluators

Evaluators are the core building blocks of Patronus's evaluation system.
This page covers how to create and use your own custom evaluators to assess LLM outputs according to your specific criteria.

## Creating Basic Evaluators

The simplest way to create an evaluator is with the `@evaluator()` decorator:

```python
from patronus import evaluator

@evaluator()
def keyword_match(text: str, keywords: list[str]) -> float:
    """
    Evaluates whether the text contains the specified keywords.
    Returns a score between 0.0 and 1.0 based on the percentage of matched keywords.
    """
    matches = sum(keyword.lower() in text.lower() for keyword in keywords)
    return matches / len(keywords) if keywords else 0.0
```

This decorator automatically:

- Integrates with the Patronus tracing
- Exports evaluation results to the Patronus Platform

### Flexible Input and Output

User-defined evaluators can accept any parameters and return several types of results:

```python
# Boolean evaluator (pass/fail)
@evaluator()
def contains_answer(text: str, answer: str) -> bool:
    return answer.lower() in text.lower()


# Numeric evaluator (score)
@evaluator()
def semantic_similarity(text1: str, text2: str) -> float:
    # Simple example - in practice use proper semantic similarity
    words1, words2 = set(text1.lower().split()), set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0


# String evaluator
@evaluator()
def tone_classifier(text: str) -> str:
    positive = ['good', 'excellent', 'great', 'helpful']
    negative = ['bad', 'poor', 'unhelpful', 'wrong']

    pos_count = sum(word in text.lower() for word in positive)
    neg_count = sum(word in text.lower() for word in negative)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"
```

### Return Types

Evaluators can return different types which are automatically converted to `EvaluationResult` objects:

- **Boolean**: `True`/`False` indicating pass/fail
- **Float/Integer**: Numerical scores (typically between 0-1)
- **String**: Text output categorizing the result
- **EvaluationResult**: Complete evaluation with scores, explanations, etc.

## Using EvaluationResult

For more detailed evaluations, return an `EvaluationResult` object:

```python
from patronus import evaluator
from patronus.evals import EvaluationResult

@evaluator()
def comprehensive_evaluation(response: str, reference: str) -> EvaluationResult:
    # Example implementation - replace with actual logic
    has_keywords = all(word in response.lower() for word in ["important", "key", "concept"])
    accuracy = 0.85  # Calculated accuracy score

    return EvaluationResult(
        score=accuracy,  # Numeric score (typically 0-1)
        pass_=accuracy >= 0.7,  # Boolean pass/fail
        text_output="Satisfactory" if accuracy >= 0.7 else "Needs improvement",  # Category
        explanation=f"Response {'contains' if has_keywords else 'is missing'} key terms. Accuracy: {accuracy:.2f}",
        metadata={  # Additional structured data
            "has_required_keywords": has_keywords,
            "response_length": len(response),
            "accuracy": accuracy
        }
    )
```

The `EvaluationResult` object can include:

- **score**: Numerical assessment (typically 0-1)
- **pass_**: Boolean pass/fail status
- **text_output**: Categorical or textual result
- **explanation**: Human-readable explanation of the result
- **metadata**: Additional structured data for analysis
- **tags**: Key-value pairs for filtering and organization


## Using Evaluators

Once defined, evaluators can be used directly:

```python
# Use evaluators as normal function
result = keyword_match("The capital of France is Paris", ["capital", "France", "Paris"])
print(f"Score: {result}")  # Output: Score: 1.0


# Using class-based evaluator
safety_check = ContentSafetyEvaluator()
result = safety_check.evaluate(
    task_output="This is a helpful and safe response."
)
print(f"Safety check passed: {result.pass_}")  # Output: Safety check passed: True
```
