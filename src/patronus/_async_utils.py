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


def run_until_complete(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    return loop.run_until_complete(coro)
