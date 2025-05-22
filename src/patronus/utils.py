from threading import Lock

from typing import Optional, Callable
import warnings

NOT_GIVEN = object()


def merge_tags(tags: Optional[dict], new_tags: Optional[dict], experiment_tags: Optional[dict]) -> dict:
    tags = tags or {}
    new_tags = new_tags or {}
    experiment_tags = experiment_tags or {}

    tags = {**tags, **new_tags}
    common_keys = set(tags.keys()) & set(experiment_tags.keys())
    diff = {key for key in common_keys if tags[key] != experiment_tags[key]}
    if diff:
        warnings.warn(f"Overriding experiment tags is not allowed. Tried to override tag: {list(diff)!r}")
    return {**tags, **experiment_tags}


class Once:
    # Execute a function exactly once and block all callers until the function returns
    #
    # Same as golang's `sync.Once <https://pkg.go.dev/sync#Once>`_

    def __init__(self) -> None:
        self._lock = Lock()
        self._done = False

    def do_once(self, func: Callable[[], None]) -> bool:
        # Execute ``func`` if it hasn't been executed or return.
        #
        # Will block until ``func`` has been called by one thread.
        #
        # Returns:
        #     Whether ``func`` was executed in this call

        # fast path, try to avoid locking
        if self._done:
            return False

        with self._lock:
            if not self._done:
                func()
                self._done = True
                return True
        return False
