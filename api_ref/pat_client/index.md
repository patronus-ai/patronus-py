# Patronus Objects

## client_async

### AsyncPatronus

```python
AsyncPatronus(max_workers: int = 10)

```

Source code in `src/patronus/pat_client/client_async.py`

```python
def __init__(self, max_workers: int = 10):
    self._pending_tasks = collections.deque()
    self._executor = ThreadPoolExecutor(max_workers=max_workers)
    self._semaphore = asyncio.Semaphore(max_workers)

```

#### evaluate

```python
evaluate(
    evaluators: Union[List[Evaluator], Evaluator],
    *,
    system_prompt: Optional[str] = None,
    task_context: Union[list[str], str, None] = None,
    task_input: Optional[str] = None,
    task_output: Optional[str] = None,
    gold_answer: Optional[str] = None,
    task_metadata: Optional[dict] = None,
    return_exceptions: bool = False,
) -> EvaluationContainer

```

Run multiple evaluators in parallel.

Source code in `src/patronus/pat_client/client_async.py`

```python
async def evaluate(
    self,
    evaluators: Union[List[Evaluator], Evaluator],
    *,
    system_prompt: Optional[str] = None,
    task_context: Union[list[str], str, None] = None,
    task_input: Optional[str] = None,
    task_output: Optional[str] = None,
    gold_answer: Optional[str] = None,
    task_metadata: Optional[dict] = None,
    return_exceptions: bool = False,
) -> EvaluationContainer:
    """
    Run multiple evaluators in parallel.
    """
    singular_eval = not isinstance(evaluators, list)
    if singular_eval:
        evaluators = [evaluators]
    evaluators = self._map_evaluators(evaluators)

    def into_coro(fn, **kwargs):
        if inspect.iscoroutinefunction(fn):
            coro = fn(**kwargs)
        else:
            coro = asyncio.to_thread(fn, **kwargs)
        return with_semaphore(self._semaphore, coro)

    with bundled_eval():
        results = await asyncio.gather(
            *(
                into_coro(
                    ev.evaluate,
                    system_prompt=system_prompt,
                    task_context=task_context,
                    task_input=task_input,
                    task_output=task_output,
                    gold_answer=gold_answer,
                    task_metadata=task_metadata,
                )
                for ev in evaluators
            ),
            return_exceptions=return_exceptions,
        )
    return EvaluationContainer(results)

```

#### evaluate_bg

```python
evaluate_bg(
    evaluators: Union[List[Evaluator], Evaluator],
    *,
    system_prompt: Optional[str] = None,
    task_context: Union[list[str], str, None] = None,
    task_input: Optional[str] = None,
    task_output: Optional[str] = None,
    gold_answer: Optional[str] = None,
    task_metadata: Optional[dict] = None,
) -> Task[EvaluationContainer]

```

Run multiple evaluators in parallel. The returned task will be a background task.

Source code in `src/patronus/pat_client/client_async.py`

```python
def evaluate_bg(
    self,
    evaluators: Union[List[Evaluator], Evaluator],
    *,
    system_prompt: Optional[str] = None,
    task_context: Union[list[str], str, None] = None,
    task_input: Optional[str] = None,
    task_output: Optional[str] = None,
    gold_answer: Optional[str] = None,
    task_metadata: Optional[dict] = None,
) -> Task[EvaluationContainer]:
    """
    Run multiple evaluators in parallel. The returned task will be a background task.
    """
    loop = asyncio.get_running_loop()
    task = loop.create_task(
        self.evaluate(
            evaluators=evaluators,
            system_prompt=system_prompt,
            task_context=task_context,
            task_input=task_input,
            task_output=task_output,
            gold_answer=gold_answer,
            task_metadata=task_metadata,
            return_exceptions=True,
        ),
        name="evaluate_bg",
    )
    self._pending_tasks.append(task)
    task.add_done_callback(self._consume_tasks)
    return task

```

#### close

```python
close()

```

Gracefully close the client. This will wait for all background tasks to finish.

Source code in `src/patronus/pat_client/client_async.py`

```python
async def close(self):
    """
    Gracefully close the client. This will wait for all background tasks to finish.
    """
    while len(self._pending_tasks) != 0:
        await self._pending_tasks.popleft()

```

## client_sync

### Patronus

```python
Patronus(workers: int = 10, shutdown_on_exit: bool = True)

```

Source code in `src/patronus/pat_client/client_sync.py`

```python
def __init__(self, workers: int = 10, shutdown_on_exit: bool = True):
    self._worker_pool = ThreadPool(workers)
    self._supervisor_pool = ThreadPool(workers)

    self._at_exit_handler = None
    if shutdown_on_exit:
        self._at_exit_handler = atexit.register(self.close)

```

#### evaluate

```python
evaluate(
    evaluators: Union[list[Evaluator], Evaluator],
    *,
    system_prompt: Optional[str] = None,
    task_context: Union[list[str], str, None] = None,
    task_input: Optional[str] = None,
    task_output: Optional[str] = None,
    gold_answer: Optional[str] = None,
    task_metadata: Optional[dict[str, Any]] = None,
    return_exceptions: bool = False,
) -> EvaluationContainer

```

Run multiple evaluators in parallel.

Source code in `src/patronus/pat_client/client_sync.py`

```python
def evaluate(
    self,
    evaluators: typing.Union[list[Evaluator], Evaluator],
    *,
    system_prompt: typing.Optional[str] = None,
    task_context: typing.Union[list[str], str, None] = None,
    task_input: typing.Optional[str] = None,
    task_output: typing.Optional[str] = None,
    gold_answer: typing.Optional[str] = None,
    task_metadata: typing.Optional[dict[str, typing.Any]] = None,
    return_exceptions: bool = False,
) -> EvaluationContainer:
    """
    Run multiple evaluators in parallel.
    """
    if not isinstance(evaluators, list):
        evaluators = [evaluators]
    evaluators = self._map_evaluators(evaluators)

    with bundled_eval():
        callables = [
            _into_thread_run_fn(
                ev.evaluate,
                system_prompt=system_prompt,
                task_context=task_context,
                task_input=task_input,
                task_output=task_output,
                gold_answer=gold_answer,
                task_metadata=task_metadata,
            )
            for ev in evaluators
        ]
        results = self._process_batch(callables, return_exceptions=return_exceptions)
        return EvaluationContainer(results)

```

#### evaluate_bg

```python
evaluate_bg(
    evaluators: list[StructuredEvaluator],
    *,
    system_prompt: Optional[str] = None,
    task_context: Union[list[str], str, None] = None,
    task_input: Optional[str] = None,
    task_output: Optional[str] = None,
    gold_answer: Optional[str] = None,
    task_metadata: Optional[dict[str, Any]] = None,
) -> TypedAsyncResult[EvaluationContainer]

```

Run multiple evaluators in parallel. The returned task will be a background task.

Source code in `src/patronus/pat_client/client_sync.py`

```python
def evaluate_bg(
    self,
    evaluators: list[StructuredEvaluator],
    *,
    system_prompt: typing.Optional[str] = None,
    task_context: typing.Union[list[str], str, None] = None,
    task_input: typing.Optional[str] = None,
    task_output: typing.Optional[str] = None,
    gold_answer: typing.Optional[str] = None,
    task_metadata: typing.Optional[dict[str, typing.Any]] = None,
) -> TypedAsyncResult[EvaluationContainer]:
    """
    Run multiple evaluators in parallel. The returned task will be a background task.
    """

    def _run():
        with bundled_eval():
            callables = [
                _into_thread_run_fn(
                    ev.evaluate,
                    system_prompt=system_prompt,
                    task_context=task_context,
                    task_input=task_input,
                    task_output=task_output,
                    gold_answer=gold_answer,
                    task_metadata=task_metadata,
                )
                for ev in evaluators
            ]
            results = self._process_batch(callables, return_exceptions=True)
            return EvaluationContainer(results)

    return typing.cast(
        TypedAsyncResult[EvaluationContainer], self._supervisor_pool.apply_async(_into_thread_run_fn(_run))
    )

```

#### close

```python
close()

```

Gracefully close the client. This will wait for all background tasks to finish.

Source code in `src/patronus/pat_client/client_sync.py`

```python
def close(self):
    """
    Gracefully close the client. This will wait for all background tasks to finish.
    """
    self._close()
    if self._at_exit_handler:
        atexit.unregister(self._at_exit_handler)

```

## container

### EvaluationContainer

```python
EvaluationContainer(
    results: list[Union[EvaluationResult, None, Exception]],
)

```

#### format

```python
format() -> str

```

Format the evaluation results into a readable summary.

Source code in `src/patronus/pat_client/container.py`

```python
def format(self) -> str:
    """
    Format the evaluation results into a readable summary.
    """
    buf = StringIO()

    total = len(self.results)
    exceptions_count = sum(1 for r in self.results if isinstance(r, Exception))
    successes_count = sum(1 for r in self.results if isinstance(r, EvaluationResult) and r.pass_ is True)
    failures_count = sum(1 for r in self.results if isinstance(r, EvaluationResult) and r.pass_ is False)

    buf.write(f"Total evaluations: {total}\n")
    buf.write(f"Successes: {successes_count}\n")
    buf.write(f"Failures: {failures_count}\n")
    buf.write(f"Exceptions: {exceptions_count}\n\n")
    buf.write("Evaluation Details:\n")
    buf.write("---\n")

    # Add detailed evaluation results
    for result in self.results:
        if result is None:
            buf.write("None\n")
        elif isinstance(result, Exception):
            buf.write(str(result))
            buf.write("\n")
        else:
            buf.write(result.format())
        buf.write("---\n")

    return buf.getvalue()

```

#### pretty_print

```python
pretty_print(file: Optional[IO] = None) -> None

```

Formats and prints the current object in a human-readable form.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file` | `Optional[IO]` | | `None` |

Source code in `src/patronus/pat_client/container.py`

```python
def pretty_print(self, file: Optional[IO] = None) -> None:
    """
    Formats and prints the current object in a human-readable form.

    Args:
        file:
    """
    f = self.format()
    print(f, file=file)

```

#### has_exception

```python
has_exception() -> bool

```

Checks if the results contain any exception.

Source code in `src/patronus/pat_client/container.py`

```python
def has_exception(self) -> bool:
    """
    Checks if the results contain any exception.
    """
    return any(isinstance(r, Exception) for r in self.results)

```

#### raise_on_exception

```python
raise_on_exception() -> None

```

Checks the results for any exceptions and raises them accordingly.

Source code in `src/patronus/pat_client/container.py`

```python
def raise_on_exception(self) -> None:
    """
    Checks the results for any exceptions and raises them accordingly.
    """
    if not self.has_exception():
        return None
    exceptions = list(r for r in self.results if isinstance(r, Exception))
    if len(exceptions) == 1:
        raise exceptions[0]
    raise MultiException(exceptions)

```

#### all_succeeded

```python
all_succeeded(ignore_exceptions: bool = False) -> bool

```

Check if all evaluations that were actually evaluated passed.

Evaluations are only considered if they:

- Have a non-None pass\_ flag set
- Are not None (skipped)
- Are not exceptions (unless ignore_exceptions=True)

Note: Returns True if no evaluations met the above criteria (empty case).

Source code in `src/patronus/pat_client/container.py`

```python
def all_succeeded(self, ignore_exceptions: bool = False) -> bool:
    """
    Check if all evaluations that were actually evaluated passed.

    Evaluations are only considered if they:
    - Have a non-None pass_ flag set
    - Are not None (skipped)
    - Are not exceptions (unless ignore_exceptions=True)

    Note: Returns True if no evaluations met the above criteria (empty case).
    """
    for r in self.results:
        if isinstance(r, Exception) and not ignore_exceptions:
            self.raise_on_exception()
        if r is not None and r.pass_ is False:
            return False
    return True

```

#### any_failed

```python
any_failed(ignore_exceptions: bool = False) -> bool

```

Check if any evaluation that was actually evaluated failed.

Evaluations are only considered if they:

- Have a non-None pass\_ flag set
- Are not None (skipped)
- Are not exceptions (unless ignore_exceptions=True)

Note: Returns False if no evaluations met the above criteria (empty case).

Source code in `src/patronus/pat_client/container.py`

```python
def any_failed(self, ignore_exceptions: bool = False) -> bool:
    """
    Check if any evaluation that was actually evaluated failed.

    Evaluations are only considered if they:
    - Have a non-None pass_ flag set
    - Are not None (skipped)
    - Are not exceptions (unless ignore_exceptions=True)

    Note: Returns False if no evaluations met the above criteria (empty case).
    """
    for r in self.results:
        if isinstance(r, Exception) and not ignore_exceptions:
            self.raise_on_exception()
        if r is not None and r.pass_ is False:
            return True
    return False

```

#### failed_evaluations

```python
failed_evaluations() -> Generator[
    EvaluationResult, None, None
]

```

Generates all failed evaluations from the results.

Source code in `src/patronus/pat_client/container.py`

```python
def failed_evaluations(self) -> Generator[EvaluationResult, None, None]:
    """
    Generates all failed evaluations from the results.
    """
    return (r for r in self.results if not isinstance(r, (Exception, type(None))) and r.pass_ is False)

```

#### succeeded_evaluations

```python
succeeded_evaluations() -> Generator[
    EvaluationResult, None, None
]

```

Generates all successfully passed evaluations from the `results` attribute.

Source code in `src/patronus/pat_client/container.py`

```python
def succeeded_evaluations(self) -> Generator[EvaluationResult, None, None]:
    """
    Generates all successfully passed evaluations from the `results` attribute.
    """
    return (r for r in self.results if not isinstance(r, (Exception, type(None))) and r.pass_ is True)

```
