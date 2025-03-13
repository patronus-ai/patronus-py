import dataclasses
from typing import Union, Generator

from patronus.evals import EvaluationResult
from patronus.exceptions import MultiException


@dataclasses.dataclass
class EvaluationContainer:
    results: list[Union[EvaluationResult, None, Exception]]

    def has_exception(self) -> bool:
        return any(isinstance(r, Exception) for r in self.results)

    def raise_on_exception(self) -> None:
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
        return (r for r in self.results if not isinstance(r, (Exception, type(None))) and r.pass_ is False)

    def succeeded_evaluations(self) -> Generator[EvaluationResult, None, None]:
        return (r for r in self.results if not isinstance(r, (Exception, type(None))) and r.pass_ is True)
