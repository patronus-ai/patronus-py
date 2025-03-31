import atexit
import collections
import itertools
import logging
import os
import threading
import time
import typing

from patronus.api import PatronusAPIClient, api_types
from patronus.utils import Once

logger = logging.getLogger("patronus.core")

_BEE_RESET_ONCE = Once()


# _FlushRequest is based on _FlushRequest from otel sdk trace batch processor/exporter.
#
# Represents a request for the BatchEvaluationExporter to flush evaluations.
class _FlushRequest:
    __slots__ = ["event", "num_evaluations"]

    def __init__(self):
        self.event = threading.Event()
        self.num_evaluations = 0


def backoff(max_value: typing.Optional[int] = None):
    for i in itertools.count(0):
        out = 2**i
        yield min(out, max_value) if max_value else out


# BatchEvaluationExporter is based by otel sdk trace batch processor.
class BatchEvaluationExporter:
    _client: PatronusAPIClient
    done: bool
    schedule_delay: float
    max_queue_size: int
    queue: typing.Deque[api_types.ClientEvaluation]
    condition: threading.Condition
    max_export_batch_size: int
    evaluations_list: list[typing.Optional[api_types.ClientEvaluation]]
    flush_request: typing.Optional[_FlushRequest]
    _pid: int
    _atexit_handler: typing.Optional[typing.Callable[[], None]]

    def __init__(
        self,
        client: PatronusAPIClient,
        *,
        schedule_delay: float = 5.0,
        max_export_batch_size: int = 100,
        max_queue_size: int = 2000,
        shutdown_on_exit: bool = True,
    ):
        self._client = client

        self.done = False
        self.schedule_delay = schedule_delay
        self.max_queue_size = max_queue_size
        self.queue = collections.deque(maxlen=self.max_queue_size)
        self.condition = threading.Condition(threading.Lock())

        self.max_export_batch_size = max_export_batch_size
        # precallocated list to export evaluations
        self.evaluations_list = [None] * self.max_export_batch_size
        self.flush_request = None

        self.worker_thread = threading.Thread(name="PatronusBatchEvaluationExporter", target=self.worker, daemon=True)
        self.worker_thread.start()

        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._at_fork_reinit)  # pylint: disable=protected-access
        self._pid = os.getpid()

        self._atexit_handler = None
        if shutdown_on_exit:
            self._atexit_handler = atexit.register(self.shutdown)

    def submit(self, data: api_types.ClientEvaluation):
        if self.done:
            logger.warning("Already shutdown, dropping evaluation.")
            return

        if self._pid != os.getpid():
            _BEE_RESET_ONCE.do_once(self._at_fork_reinit)

        if len(self.queue) >= self.max_queue_size:
            logger.info("Queue is full, likely evaluations will be dropped.")

        self.queue.appendleft(data)

        if len(self.queue) >= self.max_export_batch_size:
            with self.condition:
                self.condition.notify()

    def _at_fork_reinit(self):
        self.condition = threading.Condition(threading.Lock())
        self.queue.clear()

        # worker_thread is local to a process, only the thread that issued fork continues
        # to exist. A new worker thread must be started in child process.
        self.worker_thread = threading.Thread(name="PatronusBatchEvaluationExporter", target=self.worker, daemon=True)
        self.worker_thread.start()
        self._pid = os.getpid()

    def worker(self):
        timeout = self.schedule_delay
        flush_request: typing.Optional[_FlushRequest] = None

        while not self.done:
            with self.condition:
                if self.done:
                    # done flag may have changed, avoid waiting
                    break

                flush_request = self._get_and_unset_flush_request()
                if (len(self.queue) < self.max_export_batch_size) and flush_request is None:
                    self.condition.wait(timeout)
                    flush_request = self._get_and_unset_flush_request()

                if not self.queue:
                    timeout = self.schedule_delay
                    self._notify_flush_request_finished(flush_request)
                    flush_request = None
                    continue

                if self.done:
                    break

            start = time.time_ns()
            self._export(flush_request)
            end = time.time_ns()
            elapsed = (end - start) / 1e9
            timeout = self.schedule_delay - elapsed

            self._notify_flush_request_finished(flush_request)
            flush_request = None

        # there might have been a new flush request while export was running
        # and before the done flag switched to true
        with self.condition:
            shutdown_flush_request = self._get_and_unset_flush_request()

        self._drain_queue()
        self._notify_flush_request_finished(flush_request)
        self._notify_flush_request_finished(shutdown_flush_request)

    def _drain_queue(self) -> int:
        exported = 0
        while self.queue:
            exported += self._export_batch()
        return exported

    def _export(self, flush_request: typing.Optional[_FlushRequest] = None) -> None:
        if not flush_request:
            self._export_batch()
            return

        num_evals = flush_request.num_evaluations
        while self.queue:
            num_exported = self._export_batch()
            num_evals -= num_exported

            if num_evals <= 0:
                break

    def _export_batch(self) -> int:
        idx = 0
        while idx < self.max_export_batch_size and self.queue:
            self.evaluations_list[idx] = self.queue.pop()
            idx += 1

        try:
            self.__export_batch(self.evaluations_list[:idx])
        except Exception:
            logger.exception("Failed to export batch, evaluations are dropped most likely.")

        for index in range(idx):
            self.evaluations_list[index] = None

        return idx

    def __export_batch(self, evaluations: typing.List[api_types.ClientEvaluation]) -> None:
        max_delay = 64
        last_exc = None
        for delay in backoff(max_value=max_delay):
            try:
                self._client.batch_create_evaluations_sync(
                    request=api_types.BatchCreateEvaluationsRequest(evaluations=evaluations)
                )
                return
            except Exception as e:
                logger.exception(f"Failed to export evaluations. retrying in {delay} seconds.")
                time.sleep(delay)
                last_exc = e

        if last_exc:
            raise last_exc

    def _get_and_unset_flush_request(self) -> typing.Optional[_FlushRequest]:
        flush_request = self.flush_request
        self.flush_request = None
        if flush_request is not None:
            flush_request.num_evaluations = len(self.queue)
        return flush_request

    @staticmethod
    def _notify_flush_request_finished(flush_request: typing.Optional[_FlushRequest]) -> None:
        if flush_request is not None:
            flush_request.event.set()

    def _get_or_create_flush_request(self) -> _FlushRequest:
        if self.flush_request is None:
            self.flush_request = _FlushRequest()
        return self.flush_request

    def force_flush(self, timeout: typing.Optional[float] = None) -> bool:
        timeout = timeout or self.schedule_delay

        if self.done:
            logger.warning("Already shutdown, ignoring call to force_flush().")
            return True

        with self.condition:
            flush_request = self._get_or_create_flush_request()
            # signal the worker thread to flush and wait for it to finish
            self.condition.notify_all()

        # wait for token to be processed
        ret = flush_request.event.wait(timeout)
        if not ret:
            logger.warning("Timeout was exceeded in force_flush().")
        return ret

    def shutdown(self):
        self._shutdown()
        if self._atexit_handler:
            atexit.unregister(self._atexit_handler)

    def _shutdown(self):
        self.done = True
        with self.condition:
            self.condition.notify_all()
        self.worker_thread.join()
