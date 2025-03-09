import logging

import asyncio
import datetime
import functools
import inspect
import random
import time
import traceback
from typing import Optional

from patronus.api.api_client_base import RetryError, RPMLimitError, UnrecoverableAPIError

log = logging.getLogger("patronus.core")


def retry(max_attempts=3, initial_delay=1, backoff_factor=2):
    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                attempts = 1
                delay = initial_delay
                last_error = None
                stack_trace: Optional[str] = None

                while attempts <= max_attempts:
                    call_time = datetime.datetime.now()
                    try:
                        return await func(*args, **kwargs)
                    except UnrecoverableAPIError as e:
                        raise RetryError(attempts, max_attempts, e, traceback.format_exc())
                    except RPMLimitError as err:
                        last_error = err
                        stack_trace = traceback.format_exc()
                        log.debug(f"api_retry: Attempt {attempts} out of {max_attempts}: {err}")
                        wait_for_s = err.wait_for_s
                        if not wait_for_s:
                            now = datetime.datetime.now()
                            if call_time.minute != now.minute:
                                continue
                            next_minute = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
                            wait_for_s = (next_minute - now).total_seconds() + (3 + random.random() * 5)
                        if attempts < max_attempts:
                            log.debug(f"Attempting again in {wait_for_s}s.")
                            await asyncio.sleep(wait_for_s)
                    except Exception as err:
                        log.debug(f"api_retry: Attempt {attempts} out of {max_attempts}: {err}")
                        last_error = err
                        stack_trace = traceback.format_exc()
                        if attempts < max_attempts:
                            await asyncio.sleep(delay)
                    finally:
                        delay *= backoff_factor
                        attempts += 1

                raise RetryError(attempts - 1, max_attempts, last_error, stack_trace) from last_error

            return wrapper
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                attempts = 1
                delay = initial_delay
                last_error = None
                stack_trace: Optional[str] = None

                while attempts <= max_attempts:
                    call_time = datetime.datetime.now()
                    try:
                        return func(*args, **kwargs)
                    except UnrecoverableAPIError as e:
                        raise RetryError(attempts, max_attempts, e, traceback.format_exc())
                    except RPMLimitError as err:
                        last_error = err
                        stack_trace = traceback.format_exc()
                        log.debug(f"api_retry: Attempt {attempts} out of {max_attempts}: {err}")
                        wait_for_s = err.wait_for_s
                        if not wait_for_s:
                            now = datetime.datetime.now()
                            if call_time.minute != now.minute:
                                continue
                            next_minute = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
                            wait_for_s = (next_minute - now).total_seconds() + (3 + random.random() * 5)
                        if attempts < max_attempts:
                            log.debug(f"Attempting again in {wait_for_s}s.")
                            time.sleep(wait_for_s)
                    except Exception as err:
                        log.debug(f"api_retry: Attempt {attempts} out of {max_attempts}: {err}")
                        last_error = err
                        stack_trace = traceback.format_exc()
                        if attempts < max_attempts:
                            time.sleep(delay)
                    finally:
                        delay *= backoff_factor
                        attempts += 1

                raise RetryError(attempts - 1, max_attempts, last_error, stack_trace) from last_error

            return wrapper

    return decorator
