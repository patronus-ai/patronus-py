import asyncio
import functools
import inspect
import typing
from concurrent.futures import ThreadPoolExecutor

T = typing.TypeVar("T")


async def run_as_coro(
    __loop: asyncio.AbstractEventLoop,
    __executor: ThreadPoolExecutor,
    __fn: typing.Callable[[typing.Any, ...], T],
    *args,
    **kwargs,
) -> typing.Awaitable[T]:
    if inspect.iscoroutinefunction(__fn):
        return await __fn(*args, **kwargs)

    @functools.wraps(__fn)
    def inner(args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any]):
        return __fn(*args, **kwargs)

    return await __loop.run_in_executor(__executor, inner, args, kwargs)


def run_until_complete(coro: typing.Awaitable[T]) -> typing.Awaitable[T]:
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        # Ignore this exception since we can continue by creating an event loop.
        # We are not creating an event loop in the except block to no include
        # this stacktrace part in case of a raised exception in the coroutine.
        # Otherwise, we would get message as
        #
        # > RuntimeError: no running event loop
        # > During handling of the above exception, another exception occurred: [...]
        #
        # Which we don't want because the try catch is just a check whether there is a running event loop.
        # We are not really handling an "exception" here.
        pass

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)
