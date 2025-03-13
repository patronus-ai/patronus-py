from typing import Optional


class WithStackTraceMixin:
    stack_trace: Optional[str] = None


class MultiException(Exception):
    def __init__(self, exceptions: list[Exception]):
        self.exceptions = exceptions
        super().__init__(f"Multiple exception occurred: {self.exceptions}")


class UninitializedError(Exception): ...
