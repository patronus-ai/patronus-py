import asyncio
from sys import version_info

from tqdm.asyncio import tqdm as tqdm_async

from typing_extensions import Self


class AsyncTQDMWithHandle(tqdm_async):
    # Workaround for accessing tqdm instance with async tasks.
    # Instead of calling gather which don't provide access to tqdm instance:
    # ```
    # tqdm_async.gather(features)
    # ```

    # Call prep_gather() follow by gather()
    # ```
    # tqdm_instance = AsyncTQDMWithHandle.pre_gather(features)
    # ...
    # tqdm_instance.gather()
    # ```

    # tqdm_instance can be used to clear and display progress bar using tqdm_instance.clear() and
    # tqdm_instance.display() methods.

    async def gather(self):
        def into_iter():
            yield from self

        res = [await f for f in into_iter()]
        return [i for _, i in sorted(res)]

    @classmethod
    async def prep_gather(cls, *fs, loop=None, timeout=None, total=None, **tqdm_kwargs) -> Self:
        async def wrap_awaitable(i, f):
            return i, await f

        ifs = [wrap_awaitable(i, f) for i, f in enumerate(fs)]
        return cls.prep_as_completed(ifs, loop=loop, timeout=timeout, total=total, **tqdm_kwargs)

    @classmethod
    def prep_as_completed(cls, fs, *, loop=None, timeout=None, total=None, **tqdm_kwargs):
        if total is None:
            total = len(fs)
        kwargs = {}
        if version_info[:2] < (3, 10):
            kwargs["loop"] = loop
        return cls(
            asyncio.as_completed(fs, timeout=timeout, **kwargs),
            total=total,
            **tqdm_kwargs,
        )
