import dataclasses
from typing import Union, Generator, IO, Optional
from io import StringIO

from patronus.evals import EvaluationResult
from patronus.exceptions import MultiException


@dataclasses.dataclass
class EvaluationContainer:
    results: list[Union[EvaluationResult, None, Exception]]

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

    def pretty_print(self, file: Optional[IO] = None) -> None:
        """
        Formats and prints the current object in a human-readable form.

        Args:
            file:
        """
        f = self.format()
        print(f, file=file)

    def has_exception(self) -> bool:
        """
        Checks if the results contain any exception.
        """
        return any(isinstance(r, Exception) for r in self.results)

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

    def failed_evaluations(self) -> Generator[EvaluationResult, None, None]:
        """
        Generates all failed evaluations from the results.
        """
        return (r for r in self.results if not isinstance(r, (Exception, type(None))) and r.pass_ is False)

    def succeeded_evaluations(self) -> Generator[EvaluationResult, None, None]:
        """
        Generates all successfully passed evaluations from the `results` attribute.
        """
        return (r for r in self.results if not isinstance(r, (Exception, type(None))) and r.pass_ is True)
