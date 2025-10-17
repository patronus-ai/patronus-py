# evals

## patronus.evals

### evaluators

#### Evaluator

```python
Evaluator(weight: Optional[Union[str, float]] = None)
```

Base Evaluator Class

Source code in `src/patronus/evals/evaluators.py`

```python
def __init__(self, weight: Optional[Union[str, float]] = None):
    if weight is not None:
        try:
            decimal.Decimal(str(weight))
        except (decimal.InvalidOperation, ValueError, TypeError):
            raise TypeError(
                f"{weight} is not a valid weight. Weight must be a valid decimal number (string or float)."
            )
    self.weight = weight
```

##### evaluate

```python
evaluate(*args, **kwargs) -> Optional[EvaluationResult]
```

Synchronous version of evaluate method. When inheriting directly from Evaluator class it's permitted to change parameters signature. Return type should stay unchanged.

Source code in `src/patronus/evals/evaluators.py`

```python
@abc.abstractmethod
def evaluate(self, *args, **kwargs) -> Optional[EvaluationResult]:
    """
    Synchronous version of evaluate method.
    When inheriting directly from Evaluator class it's permitted to change parameters signature.
    Return type should stay unchanged.
    """
```

#### AsyncEvaluator

```python
AsyncEvaluator(weight: Optional[Union[str, float]] = None)
```

Bases: `Evaluator`

Source code in `src/patronus/evals/evaluators.py`

```python
def __init__(self, weight: Optional[Union[str, float]] = None):
    if weight is not None:
        try:
            decimal.Decimal(str(weight))
        except (decimal.InvalidOperation, ValueError, TypeError):
            raise TypeError(
                f"{weight} is not a valid weight. Weight must be a valid decimal number (string or float)."
            )
    self.weight = weight
```

##### evaluate

```python
evaluate(*args, **kwargs) -> Optional[EvaluationResult]
```

Asynchronous version of evaluate method. When inheriting directly from Evaluator class it's permitted to change parameters signature. Return type should stay unchanged.

Source code in `src/patronus/evals/evaluators.py`

```python
@abc.abstractmethod
async def evaluate(self, *args, **kwargs) -> Optional[EvaluationResult]:
    """
    Asynchronous version of evaluate method.
    When inheriting directly from Evaluator class it's permitted to change parameters signature.
    Return type should stay unchanged.
    """
```

#### StructuredEvaluator

```python
StructuredEvaluator(weight: Optional[Union[str, float]] = None)
```

Bases: `Evaluator`

Base for structured evaluators

Source code in `src/patronus/evals/evaluators.py`

```python
def __init__(self, weight: Optional[Union[str, float]] = None):
    if weight is not None:
        try:
            decimal.Decimal(str(weight))
        except (decimal.InvalidOperation, ValueError, TypeError):
            raise TypeError(
                f"{weight} is not a valid weight. Weight must be a valid decimal number (string or float)."
            )
    self.weight = weight
```

#### AsyncStructuredEvaluator

```python
AsyncStructuredEvaluator(weight: Optional[Union[str, float]] = None)
```

Bases: `AsyncEvaluator`

Base for async structured evaluators

Source code in `src/patronus/evals/evaluators.py`

```python
def __init__(self, weight: Optional[Union[str, float]] = None):
    if weight is not None:
        try:
            decimal.Decimal(str(weight))
        except (decimal.InvalidOperation, ValueError, TypeError):
            raise TypeError(
                f"{weight} is not a valid weight. Weight must be a valid decimal number (string or float)."
            )
    self.weight = weight
```

#### RemoteEvaluatorMixin

```python
RemoteEvaluatorMixin(evaluator_id_or_alias: str, criteria: Optional[str] = None, *, tags: Optional[dict[str, str]] = None, explain_strategy: Literal['never', 'on-fail', 'on-success', 'always'] = 'always', criteria_config: Optional[dict[str, Any]] = None, allow_update: bool = False, max_attempts: int = 3, api_: Optional[PatronusAPIClient] = None, weight: Optional[Union[str, float]] = None, retry_max_attempts: Optional[int] = 3, retry_initial_delay: Optional[int] = 1, retry_backoff_factor: Optional[int] = 2)
```

Parameters:

| Name                    | Type                                                  | Description                                                                                                                                                                                                                                                                                     | Default    |
| ----------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `evaluator_id_or_alias` | `str`                                                 | The ID or alias of the evaluator to use.                                                                                                                                                                                                                                                        | *required* |
| `criteria`              | `Optional[str]`                                       | The criteria name to use for evaluation. If not provided, the evaluator's default criteria will be used.                                                                                                                                                                                        | `None`     |
| `tags`                  | `Optional[dict[str, str]]`                            | Optional tags to attach to evaluations.                                                                                                                                                                                                                                                         | `None`     |
| `explain_strategy`      | `Literal['never', 'on-fail', 'on-success', 'always']` | When to generate explanations for evaluations. Options are "never", "on-fail", "on-success", or "always".                                                                                                                                                                                       | `'always'` |
| `criteria_config`       | `Optional[dict[str, Any]]`                            | Configuration for the criteria. (Currently unused)                                                                                                                                                                                                                                              | `None`     |
| `allow_update`          | `bool`                                                | Whether to allow updates. (Currently unused)                                                                                                                                                                                                                                                    | `False`    |
| `max_attempts`          | `int`                                                 | Maximum number of retry attempts. (Currently unused)                                                                                                                                                                                                                                            | `3`        |
| `api_`                  | `Optional[PatronusAPIClient]`                         | Optional API client instance. If not provided, will use the default client from context.                                                                                                                                                                                                        | `None`     |
| `weight`                | `Optional[Union[str, float]]`                         | Optional weight for the evaluator. This is only used within the Patronus Experimentation Framework to indicate the relative importance of evaluators. Must be a valid decimal number (string or float). Weights are stored as experiment metadata and do not affect standalone evaluator usage. | `None`     |
| `retry_max_attempts`    | `Optional[int]`                                       | Maximum number of retry attempts.                                                                                                                                                                                                                                                               | `3`        |
| `retry_initial_delay`   | `Optional[int]`                                       | Initial delay before next retry.                                                                                                                                                                                                                                                                | `1`        |
| `retry_backoff_factor`  | `Optional[int]`                                       | Delay factor between retry attempts.                                                                                                                                                                                                                                                            | `2`        |

Source code in `src/patronus/evals/evaluators.py`

```python
def __init__(
    self,
    evaluator_id_or_alias: str,
    criteria: Optional[str] = None,
    *,
    tags: Optional[dict[str, str]] = None,
    explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"] = "always",
    criteria_config: Optional[dict[str, typing.Any]] = None,
    allow_update: bool = False,
    max_attempts: int = 3,
    api_: Optional[PatronusAPIClient] = None,
    weight: Optional[Union[str, float]] = None,
    retry_max_attempts: Optional[int] = 3,
    retry_initial_delay: Optional[int] = 1,
    retry_backoff_factor: Optional[int] = 2,
):
    """Initialize a remote evaluator.

    Args:
        evaluator_id_or_alias: The ID or alias of the evaluator to use.
        criteria: The criteria name to use for evaluation. If not provided,
            the evaluator's default criteria will be used.
        tags: Optional tags to attach to evaluations.
        explain_strategy: When to generate explanations for evaluations.
            Options are "never", "on-fail", "on-success", or "always".
        criteria_config: Configuration for the criteria. (Currently unused)
        allow_update: Whether to allow updates. (Currently unused)
        max_attempts: Maximum number of retry attempts. (Currently unused)
        api_: Optional API client instance. If not provided, will use the
            default client from context.
        weight: Optional weight for the evaluator. This is only used within
            the Patronus Experimentation Framework to indicate the relative
            importance of evaluators. Must be a valid decimal number (string
            or float). Weights are stored as experiment metadata and do not
            affect standalone evaluator usage.
        retry_max_attempts: Maximum number of retry attempts.
        retry_initial_delay: Initial delay before next retry.
        retry_backoff_factor: Delay factor between retry attempts.
    """
    self.evaluator_id_or_alias = evaluator_id_or_alias
    self.evaluator_id = None
    self.criteria = criteria
    self.tags = tags or {}
    self.explain_strategy = explain_strategy
    self.criteria_config = criteria_config
    self.allow_update = allow_update
    self.max_attempts = max_attempts
    self._api = api_
    self._resolved = False
    self.weight = weight
    self._load_lock = threading.Lock()
    self._async_load_lock = asyncio.Lock()
    self.retry_max_attempts = retry_max_attempts
    self.retry_initial_delay = retry_initial_delay
    self.retry_backoff_factor = retry_backoff_factor
```

#### RemoteEvaluator

```python
RemoteEvaluator(evaluator_id_or_alias: str, criteria: Optional[str] = None, *, tags: Optional[dict[str, str]] = None, explain_strategy: Literal['never', 'on-fail', 'on-success', 'always'] = 'always', criteria_config: Optional[dict[str, Any]] = None, allow_update: bool = False, max_attempts: int = 3, api_: Optional[PatronusAPIClient] = None, weight: Optional[Union[str, float]] = None, retry_max_attempts: Optional[int] = 3, retry_initial_delay: Optional[int] = 1, retry_backoff_factor: Optional[int] = 2)
```

Bases: `RemoteEvaluatorMixin`, `StructuredEvaluator`

Synchronous remote evaluator

Source code in `src/patronus/evals/evaluators.py`

```python
def __init__(
    self,
    evaluator_id_or_alias: str,
    criteria: Optional[str] = None,
    *,
    tags: Optional[dict[str, str]] = None,
    explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"] = "always",
    criteria_config: Optional[dict[str, typing.Any]] = None,
    allow_update: bool = False,
    max_attempts: int = 3,
    api_: Optional[PatronusAPIClient] = None,
    weight: Optional[Union[str, float]] = None,
    retry_max_attempts: Optional[int] = 3,
    retry_initial_delay: Optional[int] = 1,
    retry_backoff_factor: Optional[int] = 2,
):
    """Initialize a remote evaluator.

    Args:
        evaluator_id_or_alias: The ID or alias of the evaluator to use.
        criteria: The criteria name to use for evaluation. If not provided,
            the evaluator's default criteria will be used.
        tags: Optional tags to attach to evaluations.
        explain_strategy: When to generate explanations for evaluations.
            Options are "never", "on-fail", "on-success", or "always".
        criteria_config: Configuration for the criteria. (Currently unused)
        allow_update: Whether to allow updates. (Currently unused)
        max_attempts: Maximum number of retry attempts. (Currently unused)
        api_: Optional API client instance. If not provided, will use the
            default client from context.
        weight: Optional weight for the evaluator. This is only used within
            the Patronus Experimentation Framework to indicate the relative
            importance of evaluators. Must be a valid decimal number (string
            or float). Weights are stored as experiment metadata and do not
            affect standalone evaluator usage.
        retry_max_attempts: Maximum number of retry attempts.
        retry_initial_delay: Initial delay before next retry.
        retry_backoff_factor: Delay factor between retry attempts.
    """
    self.evaluator_id_or_alias = evaluator_id_or_alias
    self.evaluator_id = None
    self.criteria = criteria
    self.tags = tags or {}
    self.explain_strategy = explain_strategy
    self.criteria_config = criteria_config
    self.allow_update = allow_update
    self.max_attempts = max_attempts
    self._api = api_
    self._resolved = False
    self.weight = weight
    self._load_lock = threading.Lock()
    self._async_load_lock = asyncio.Lock()
    self.retry_max_attempts = retry_max_attempts
    self.retry_initial_delay = retry_initial_delay
    self.retry_backoff_factor = retry_backoff_factor
```

##### evaluate

```python
evaluate(*, system_prompt: Optional[str] = None, task_context: Union[list[str], str, None] = None, task_attachments: Union[list[Any], None] = None, task_input: Optional[str] = None, task_output: Optional[str] = None, gold_answer: Optional[str] = None, task_metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> EvaluationResult
```

Evaluates data using remote Patronus Evaluator

Source code in `src/patronus/evals/evaluators.py`

```python
def evaluate(
    self,
    *,
    system_prompt: Optional[str] = None,
    task_context: Union[list[str], str, None] = None,
    task_attachments: Union[list[Any], None] = None,
    task_input: Optional[str] = None,
    task_output: Optional[str] = None,
    gold_answer: Optional[str] = None,
    task_metadata: Optional[typing.Dict[str, typing.Any]] = None,
    **kwargs: Any,
) -> EvaluationResult:
    """Evaluates data using remote Patronus Evaluator"""
    kws = {
        "system_prompt": system_prompt,
        "task_context": task_context,
        "task_attachments": task_attachments,
        "task_input": task_input,
        "task_output": task_output,
        "gold_answer": gold_answer,
        "task_metadata": task_metadata,
        **kwargs,
    }
    log_id = get_current_log_id(bound_arguments=kws)

    attrs = get_context_evaluation_attributes()
    tags = {**self.tags}
    if t := attrs["tags"]:
        tags.update(t)
    tags = merge_tags(tags, kwargs.get("tags"), attrs["experiment_tags"])
    if tags:
        kws["tags"] = tags
    if did := attrs["dataset_id"]:
        kws["dataset_id"] = did
    if sid := attrs["dataset_sample_id"]:
        kws["dataset_sample_id"] = sid

    resp = retry(
        self.retry_max_attempts,
        self.retry_initial_delay,
        self.retry_backoff_factor,
    )(self._evaluate)(log_id=log_id, **kws)
    return self._translate_response(resp)
```

#### AsyncRemoteEvaluator

```python
AsyncRemoteEvaluator(evaluator_id_or_alias: str, criteria: Optional[str] = None, *, tags: Optional[dict[str, str]] = None, explain_strategy: Literal['never', 'on-fail', 'on-success', 'always'] = 'always', criteria_config: Optional[dict[str, Any]] = None, allow_update: bool = False, max_attempts: int = 3, api_: Optional[PatronusAPIClient] = None, weight: Optional[Union[str, float]] = None, retry_max_attempts: Optional[int] = 3, retry_initial_delay: Optional[int] = 1, retry_backoff_factor: Optional[int] = 2)
```

Bases: `RemoteEvaluatorMixin`, `AsyncStructuredEvaluator`

Asynchronous remote evaluator

Source code in `src/patronus/evals/evaluators.py`

```python
def __init__(
    self,
    evaluator_id_or_alias: str,
    criteria: Optional[str] = None,
    *,
    tags: Optional[dict[str, str]] = None,
    explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"] = "always",
    criteria_config: Optional[dict[str, typing.Any]] = None,
    allow_update: bool = False,
    max_attempts: int = 3,
    api_: Optional[PatronusAPIClient] = None,
    weight: Optional[Union[str, float]] = None,
    retry_max_attempts: Optional[int] = 3,
    retry_initial_delay: Optional[int] = 1,
    retry_backoff_factor: Optional[int] = 2,
):
    """Initialize a remote evaluator.

    Args:
        evaluator_id_or_alias: The ID or alias of the evaluator to use.
        criteria: The criteria name to use for evaluation. If not provided,
            the evaluator's default criteria will be used.
        tags: Optional tags to attach to evaluations.
        explain_strategy: When to generate explanations for evaluations.
            Options are "never", "on-fail", "on-success", or "always".
        criteria_config: Configuration for the criteria. (Currently unused)
        allow_update: Whether to allow updates. (Currently unused)
        max_attempts: Maximum number of retry attempts. (Currently unused)
        api_: Optional API client instance. If not provided, will use the
            default client from context.
        weight: Optional weight for the evaluator. This is only used within
            the Patronus Experimentation Framework to indicate the relative
            importance of evaluators. Must be a valid decimal number (string
            or float). Weights are stored as experiment metadata and do not
            affect standalone evaluator usage.
        retry_max_attempts: Maximum number of retry attempts.
        retry_initial_delay: Initial delay before next retry.
        retry_backoff_factor: Delay factor between retry attempts.
    """
    self.evaluator_id_or_alias = evaluator_id_or_alias
    self.evaluator_id = None
    self.criteria = criteria
    self.tags = tags or {}
    self.explain_strategy = explain_strategy
    self.criteria_config = criteria_config
    self.allow_update = allow_update
    self.max_attempts = max_attempts
    self._api = api_
    self._resolved = False
    self.weight = weight
    self._load_lock = threading.Lock()
    self._async_load_lock = asyncio.Lock()
    self.retry_max_attempts = retry_max_attempts
    self.retry_initial_delay = retry_initial_delay
    self.retry_backoff_factor = retry_backoff_factor
```

##### evaluate

```python
evaluate(*, system_prompt: Optional[str] = None, task_context: Union[list[str], str, None] = None, task_attachments: Union[list[Any], None] = None, task_input: Optional[str] = None, task_output: Optional[str] = None, gold_answer: Optional[str] = None, task_metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> EvaluationResult
```

Evaluates data using remote Patronus Evaluator

Source code in `src/patronus/evals/evaluators.py`

```python
async def evaluate(
    self,
    *,
    system_prompt: Optional[str] = None,
    task_context: Union[list[str], str, None] = None,
    task_attachments: Union[list[Any], None] = None,
    task_input: Optional[str] = None,
    task_output: Optional[str] = None,
    gold_answer: Optional[str] = None,
    task_metadata: Optional[typing.Dict[str, typing.Any]] = None,
    **kwargs: Any,
) -> EvaluationResult:
    """Evaluates data using remote Patronus Evaluator"""
    kws = {
        "system_prompt": system_prompt,
        "task_context": task_context,
        "task_attachments": task_attachments,
        "task_input": task_input,
        "task_output": task_output,
        "gold_answer": gold_answer,
        "task_metadata": task_metadata,
        **kwargs,
    }
    log_id = get_current_log_id(bound_arguments=kws)

    attrs = get_context_evaluation_attributes()
    tags = {**self.tags}
    if t := attrs["tags"]:
        tags.update(t)
    tags = merge_tags(tags, kwargs.get("tags"), attrs["experiment_tags"])
    if tags:
        kws["tags"] = tags
    if did := attrs["dataset_id"]:
        kws["dataset_id"] = did
    if sid := attrs["dataset_sample_id"]:
        kws["dataset_sample_id"] = sid

    resp = await retry(
        self.retry_max_attempts,
        self.retry_initial_delay,
        self.retry_backoff_factor,
    )(self._evaluate)(log_id=log_id, **kws)
    return self._translate_response(resp)
```

#### ensure_loading

```python
ensure_loading(func: Optional[Callable[..., Any]] = None)
```

Decorator that calls .load() on the decorated entity if the .load method exists. This ensures that remote evaluators are properly loaded before evaluation.

Source code in `src/patronus/evals/evaluators.py`

```python
def ensure_loading(
    func: Optional[typing.Callable[..., typing.Any]] = None,
):
    """
    Decorator that calls .load() on the decorated entity if the .load method exists.
    This ensures that remote evaluators are properly loaded before evaluation.
    """

    if func is None:
        return ensure_loading()(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'load') and callable(getattr(self, 'load')) and not getattr(self, '_loaded', False):
            self.load()
        return func(self, *args, **kwargs)

    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        if hasattr(self, 'load') and callable(getattr(self, 'load')) and not getattr(self, '_loaded', False):
            await self.load()
        return await func(self, *args, **kwargs)

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper
```

#### get_current_log_id

```python
get_current_log_id(bound_arguments: dict[str, Any]) -> Optional[LogID]
```

Return log_id for given arguments in current context. Returns None if there is no context - most likely SDK is not initialized.

Source code in `src/patronus/evals/evaluators.py`

```python
def get_current_log_id(bound_arguments: dict[str, Any]) -> Optional[LogID]:
    """
    Return log_id for given arguments in current context.
    Returns None if there is no context - most likely SDK is not initialized.
    """
    eval_group = _ctx_evaluation_log_group.get(None)
    if eval_group is None:
        return None
    log_id = eval_group.find_log(bound_arguments)
    if log_id is None:
        raise ValueError("Log not found for provided arguments")
    return log_id
```

#### bundled_eval

```python
bundled_eval(span_name: str = 'Evaluation bundle', attributes: Optional[dict[str, str]] = None)
```

Start a span that would automatically bundle evaluations.

Evaluations are passed by arguments passed to the evaluators called inside the context manager.

The following example would create two bundles:

- fist with arguments `x=10, y=20`
- second with arguments `spam="abc123"`

```python
with bundled_eval():
    foo_evaluator(x=10, y=20)
    bar_evaluator(x=10, y=20)
    tar_evaluator(spam="abc123")
```

Source code in `src/patronus/evals/evaluators.py`

````python
@contextlib.contextmanager
def bundled_eval(span_name: str = "Evaluation bundle", attributes: Optional[dict[str, str]] = None):
    """
    Start a span that would automatically bundle evaluations.

    Evaluations are passed by arguments passed to the evaluators called inside the context manager.

    The following example would create two bundles:

    - fist with arguments `x=10, y=20`
    - second with arguments `spam="abc123"`

    ```python
    with bundled_eval():
        foo_evaluator(x=10, y=20)
        bar_evaluator(x=10, y=20)
        tar_evaluator(spam="abc123")
    ```

    """
    tracer = context.get_tracer_or_none()
    if tracer is None:
        yield
        return

    attributes = {
        **(attributes or {}),
        Attributes.span_type.value: SpanTypes.eval.value,
    }
    with tracer.start_as_current_span(span_name, attributes=attributes):
        with _start_evaluation_log_group():
            yield
````

#### evaluator

```python
evaluator(_fn: Optional[Callable[..., Any]] = None, *, evaluator_id: Union[str, Callable[[], str], None] = None, criteria: Union[str, Callable[[], str], None] = None, metric_name: Optional[str] = None, metric_description: Optional[str] = None, is_method: bool = False, span_name: Optional[str] = None, log_none_arguments: bool = False, **kwargs: Any) -> typing.Callable[..., typing.Any]
```

Decorator for creating functional-style evaluators that log execution and results.

This decorator works with both synchronous and asynchronous functions. The decorator doesn't modify the function's return value, but records it after converting to an EvaluationResult.

Evaluators can return different types which are automatically converted to `EvaluationResult` objects:

- `bool`: `True`/`False` indicating pass/fail.
- `float`/`int`: Numerical scores (typically between 0-1).
- `str`: Text output categorizing the result.
- EvaluationResult: Complete evaluation with scores, explanations, etc.
- `None`: Indicates evaluation was skipped and no result will be recorded.

Evaluation results are exported in the background without blocking execution. The SDK must be initialized with `patronus.init()` for evaluations to be recorded, though decorated functions will still execute even without initialization.

The evaluator integrates with a context-based system to identify and handle shared evaluation logging and tracing spans.

**Example:**

```python
from patronus import init, evaluator
from patronus.evals import EvaluationResult

# Initialize the SDK to record evaluations
init()

# Simple evaluator function
@evaluator()
def exact_match(actual: str, expected: str) -> bool:
    return actual.strip() == expected.strip()

# More complex evaluator with detailed result
@evaluator()
def semantic_match(actual: str, expected: str) -> EvaluationResult:
    similarity = calculate_similarity(actual, expected)  # Your similarity function
    return EvaluationResult(
        score=similarity,
        pass_=similarity > 0.8,
        text_output="High similarity" if similarity > 0.8 else "Low similarity",
        explanation=f"Calculated similarity: {similarity}"
    )

# Use the evaluators
result = exact_match("Hello world", "Hello world")
print(f"Match: {result}")  # Output: Match: True
```

Parameters:

| Name                 | Type                                  | Description                                                                                                                                                                                                                                                                                                                                                                                     | Default |
| -------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `_fn`                | `Optional[Callable[..., Any]]`        | The function to be decorated.                                                                                                                                                                                                                                                                                                                                                                   | `None`  |
| `evaluator_id`       | `Union[str, Callable[[], str], None]` | Name for the evaluator. Defaults to function name (or class name in case of class based evaluators).                                                                                                                                                                                                                                                                                            | `None`  |
| `criteria`           | `Union[str, Callable[[], str], None]` | Name of the criteria used by the evaluator. The use of the criteria is only recommended in more complex evaluator setups where evaluation algorithm changes depending on a criteria (think strategy pattern).                                                                                                                                                                                   | `None`  |
| `metric_name`        | `Optional[str]`                       | Name for the evaluation metric. Defaults to evaluator_id value.                                                                                                                                                                                                                                                                                                                                 | `None`  |
| `metric_description` | `Optional[str]`                       | The description of the metric used for evaluation. If not provided then the docstring of the wrapped function is used for this value.                                                                                                                                                                                                                                                           | `None`  |
| `is_method`          | `bool`                                | Whether the wrapped function is a method. This value is used to determine whether to remove "self" argument from the log. It also allows for dynamic evaluator_id and criteria discovery based on get_evaluator_id() and get_criteria_id() methods. User-code usually shouldn't use it as long as user defined class-based evaluators inherit from the library provided Evaluator base classes. | `False` |
| `span_name`          | `Optional[str]`                       | Name of the span to represent this evaluation in the tracing system. Defaults to None, in which case a default name is generated based on the evaluator.                                                                                                                                                                                                                                        | `None`  |
| `log_none_arguments` | `bool`                                | Controls whether arguments with None values are included in log output. This setting affects only logging behavior and has no impact on function execution. Note: Only applies to top-level arguments. For nested structures like dictionaries, None values will always be logged regardless of this setting.                                                                                   | `False` |
| `**kwargs`           | `Any`                                 | Additional keyword arguments that may be passed to the decorator or its internal methods.                                                                                                                                                                                                                                                                                                       | `{}`    |

Returns:

| Name       | Type                 | Description                                                                                                         |
| ---------- | -------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `Callable` | `Callable[..., Any]` | Returns the decorated function with additional evaluation behavior, suitable for synchronous or asynchronous usage. |

Note

For evaluations that need to be compatible with experiments, consider using StructuredEvaluator or AsyncStructuredEvaluator classes instead.

Source code in `src/patronus/evals/evaluators.py`

````python
def evaluator(
    _fn: Optional[typing.Callable[..., typing.Any]] = None,
    *,
    evaluator_id: Union[str, typing.Callable[[], str], None] = None,
    criteria: Union[str, typing.Callable[[], str], None] = None,
    metric_name: Optional[str] = None,
    metric_description: Optional[str] = None,
    is_method: bool = False,
    span_name: Optional[str] = None,
    log_none_arguments: bool = False,
    **kwargs: typing.Any,
) -> typing.Callable[..., typing.Any]:
    """
    Decorator for creating functional-style evaluators that log execution and results.

    This decorator works with both synchronous and asynchronous functions. The decorator doesn't
    modify the function's return value, but records it after converting to an EvaluationResult.

    Evaluators can return different types which are automatically converted to `EvaluationResult` objects:

    * `bool`: `True`/`False` indicating pass/fail.
    * `float`/`int`: Numerical scores (typically between 0-1).
    * `str`: Text output categorizing the result.
    * [EvaluationResult][patronus.evals.types.EvaluationResult]: Complete evaluation with scores, explanations, etc.
    * `None`: Indicates evaluation was skipped and no result will be recorded.

    Evaluation results are exported in the background without blocking execution. The SDK must be
    initialized with `patronus.init()` for evaluations to be recorded, though decorated functions
    will still execute even without initialization.

    The evaluator integrates with a context-based system to identify and handle shared evaluation
    logging and tracing spans.

    **Example:**

    ```python
    from patronus import init, evaluator
    from patronus.evals import EvaluationResult

    # Initialize the SDK to record evaluations
    init()

    # Simple evaluator function
    @evaluator()
    def exact_match(actual: str, expected: str) -> bool:
        return actual.strip() == expected.strip()

    # More complex evaluator with detailed result
    @evaluator()
    def semantic_match(actual: str, expected: str) -> EvaluationResult:
        similarity = calculate_similarity(actual, expected)  # Your similarity function
        return EvaluationResult(
            score=similarity,
            pass_=similarity > 0.8,
            text_output="High similarity" if similarity > 0.8 else "Low similarity",
            explanation=f"Calculated similarity: {similarity}"
        )

    # Use the evaluators
    result = exact_match("Hello world", "Hello world")
    print(f"Match: {result}")  # Output: Match: True
    ```

    Args:
        _fn: The function to be decorated.
        evaluator_id: Name for the evaluator.
            Defaults to function name (or class name in case of class based evaluators).
        criteria: Name of the criteria used by the evaluator.
            The use of the criteria is only recommended in more complex evaluator setups
            where evaluation algorithm changes depending on a criteria (think strategy pattern).
        metric_name: Name for the evaluation metric. Defaults to evaluator_id value.
        metric_description: The description of the metric used for evaluation.
            If not provided then the docstring of the wrapped function is used for this value.
        is_method: Whether the wrapped function is a method.
            This value is used to determine whether to remove "self" argument from the log.
            It also allows for dynamic evaluator_id and criteria discovery
            based on `get_evaluator_id()` and `get_criteria_id()` methods.
            User-code usually shouldn't use it as long as user defined class-based evaluators inherit from
            the library provided Evaluator base classes.
        span_name: Name of the span to represent this evaluation in the tracing system.
            Defaults to None, in which case a default name is generated based on the evaluator.
        log_none_arguments: Controls whether arguments with None values are included in log output.
            This setting affects only logging behavior and has no impact on function execution.
            Note: Only applies to top-level arguments. For nested structures like dictionaries,
            None values will always be logged regardless of this setting.
        **kwargs: Additional keyword arguments that may be passed to the decorator or its internal methods.

    Returns:
        Callable: Returns the decorated function with additional evaluation behavior, suitable for
            synchronous or asynchronous usage.

    Note:
        For evaluations that need to be compatible with experiments, consider using
        [StructuredEvaluator][patronus.evals.evaluators.StructuredEvaluator] or
        [AsyncStructuredEvaluator][patronus.evals.evaluators.AsyncStructuredEvaluator] classes instead.

    """
    if _fn is not None:
        return evaluator()(_fn)

    def decorator(fn):
        fn_sign = inspect.signature(fn)

        def _get_eval_id():
            return (callable(evaluator_id) and evaluator_id()) or evaluator_id or fn.__name__

        def _get_criteria():
            return (callable(criteria) and criteria()) or criteria or None

        def _prep(*fn_args, **fn_kwargs):
            bound_args = fn_sign.bind(*fn_args, **fn_kwargs)
            arguments_to_log = _as_applied_argument(fn_sign, bound_args)
            bound_args.apply_defaults()
            self_key_name = None
            instance = None
            if is_method:
                self_key_name = next(iter(fn_sign.parameters.keys()))
                instance = bound_args.arguments[self_key_name]

            eval_id = None
            eval_criteria = None
            if isinstance(instance, Evaluator):
                eval_id = instance.get_evaluator_id()
                eval_criteria = instance.get_criteria()

            if eval_id is None:
                eval_id = _get_eval_id()
            if eval_criteria is None:
                eval_criteria = _get_criteria()

            met_name = metric_name or eval_id
            met_description = metric_description or inspect.getdoc(fn) or None

            disable_export = isinstance(instance, RemoteEvaluatorMixin) and instance._disable_export

            return PrepEval(
                span_name=span_name,
                evaluator_id=eval_id,
                criteria=eval_criteria,
                metric_name=met_name,
                metric_description=met_description,
                self_key_name=self_key_name,
                arguments=arguments_to_log,
                disable_export=disable_export,
            )

        attributes = {
            Attributes.span_type.value: SpanTypes.eval.value,
            GenAIAttributes.operation_name.value: OperationNames.eval.value,
        }

        @functools.wraps(fn)
        async def wrapper_async(*fn_args, **fn_kwargs):
            ctx = context.get_current_context_or_none()
            if ctx is None:
                return await fn(*fn_args, **fn_kwargs)

            prep = _prep(*fn_args, **fn_kwargs)

            start = time.perf_counter()
            try:
                with start_span(prep.display_name(), attributes=attributes):
                    with _get_or_start_evaluation_log_group() as log_group:
                        log_id = log_group.log(
                            logger=context.get_pat_logger(ctx),
                            is_method=is_method,
                            self_key_name=prep.self_key_name,
                            bound_arguments=prep.arguments,
                            log_none_arguments=log_none_arguments,
                        )
                        ret = await fn(*fn_args, **fn_kwargs)
            except Exception as e:
                context.get_logger(ctx).exception(f"Evaluator raised an exception: {e}")
                raise e
            if prep.disable_export:
                return ret
            elapsed = time.perf_counter() - start
            handle_eval_output(
                ctx=ctx,
                log_id=log_id,
                evaluator_id=prep.evaluator_id,
                criteria=prep.criteria,
                metric_name=prep.metric_name,
                metric_description=prep.metric_description,
                ret_value=ret,
                duration=datetime.timedelta(seconds=elapsed),
                qualname=fn.__qualname__,
            )
            return ret

        @functools.wraps(fn)
        def wrapper_sync(*fn_args, **fn_kwargs):
            ctx = context.get_current_context_or_none()
            if ctx is None:
                return fn(*fn_args, **fn_kwargs)

            prep = _prep(*fn_args, **fn_kwargs)

            start = time.perf_counter()
            try:
                with start_span(prep.display_name(), attributes=attributes):
                    with _get_or_start_evaluation_log_group() as log_group:
                        log_id = log_group.log(
                            logger=context.get_pat_logger(ctx),
                            is_method=is_method,
                            self_key_name=prep.self_key_name,
                            bound_arguments=prep.arguments,
                            log_none_arguments=log_none_arguments,
                        )
                        ret = fn(*fn_args, **fn_kwargs)
            except Exception as e:
                context.get_logger(ctx).exception(f"Evaluator raised an exception: {e}")
                raise e
            if prep.disable_export:
                return ret
            elapsed = time.perf_counter() - start
            handle_eval_output(
                ctx=ctx,
                log_id=log_id,
                evaluator_id=prep.evaluator_id,
                criteria=prep.criteria,
                metric_name=prep.metric_name,
                metric_description=prep.metric_description,
                ret_value=ret,
                duration=datetime.timedelta(seconds=elapsed),
                qualname=fn.__qualname__,
            )
            return ret

        def _set_attrs(wrapper: Any):
            wrapper._pat_evaluator = True

            # _pat_evaluator_id and _pat_criteria_id may be a bit misleading since
            # may not be correct since actually values for evaluator_id and criteria
            # are dynamically dispatched for class-based evaluators.
            # These values will be correct for function evaluators though.
            wrapper._pat_evaluator_id = _get_eval_id()
            wrapper._pat_criteria = _get_criteria()

        if inspect.iscoroutinefunction(fn):
            _set_attrs(wrapper_async)
            return wrapper_async
        else:
            _set_attrs(wrapper_sync)
            return wrapper_sync

    return decorator
````

### types

#### EvaluationResult

Bases: `BaseModel`, `LogSerializer`

Container for evaluation outcomes including score, pass/fail status, explanations, and metadata.

This class stores complete evaluation results with numeric scores, boolean pass/fail statuses, textual outputs, explanations, and arbitrary metadata. Evaluator functions can return instances of this class directly or return simpler types (bool, float, str) which will be automatically converted to EvaluationResult objects during recording.

Attributes:

| Name                   | Type                       | Description                                                                                                                           |
| ---------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `score`                | `Optional[float]`          | Score of the evaluation. Can be any numerical value, though typically ranges from 0 to 1, where 1 represents the best possible score. |
| `pass_`                | `Optional[bool]`           | Whether the evaluation is considered to pass or fail.                                                                                 |
| `text_output`          | `Optional[str]`            | Text output of the evaluation. Usually used for discrete human-readable category evaluation or as a label for score value.            |
| `metadata`             | `Optional[dict[str, Any]]` | Arbitrary json-serializable metadata about evaluation.                                                                                |
| `explanation`          | `Optional[str]`            | Human-readable explanation of the evaluation.                                                                                         |
| `tags`                 | `Optional[dict[str, str]]` | Key-value pair metadata.                                                                                                              |
| `dataset_id`           | `Optional[str]`            | ID of the dataset associated with evaluated sample.                                                                                   |
| `dataset_sample_id`    | `Optional[str]`            | ID of the sample in a dataset associated with evaluated sample.                                                                       |
| `evaluation_duration`  | `Optional[timedelta]`      | Duration of the evaluation. In case value is not set, @evaluator decorator and Evaluator classes will set this value automatically.   |
| `explanation_duration` | `Optional[timedelta]`      | Duration of the evaluation explanation.                                                                                               |

##### dump_as_log

```python
dump_as_log() -> dict[str, Any]
```

Serialize the EvaluationResult into a dictionary format suitable for logging.

Returns:

| Type             | Description                                                                  |
| ---------------- | ---------------------------------------------------------------------------- |
| `dict[str, Any]` | A dictionary containing all evaluation result fields, excluding None values. |

Source code in `src/patronus/evals/types.py`

```python
def dump_as_log(self) -> dict[str, Any]:
    """
    Serialize the EvaluationResult into a dictionary format suitable for logging.

    Returns:
        A dictionary containing all evaluation result fields, excluding None values.
    """
    return self.model_dump(mode='json')
```

##### format

```python
format() -> str
```

Format the evaluation result into a readable summary.

Source code in `src/patronus/evals/types.py`

```python
def format(self) -> str:
    """
    Format the evaluation result into a readable summary.
    """
    md = self.model_dump(exclude_none=True, mode="json")
    return yaml.dump(md)
```

##### pretty_print

```python
pretty_print(file=None) -> None
```

Pretty prints the formatted content to the specified file or standard output.

Source code in `src/patronus/evals/types.py`

```python
def pretty_print(self, file=None) -> None:
    """
    Pretty prints the formatted content to the specified file or standard output.
    """
    f = self.format()
    print(f, file=file)
```
