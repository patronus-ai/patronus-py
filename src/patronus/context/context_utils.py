import contextlib
import contextvars
import threading
import typing
from typing import Generic

import typing_extensions

T = typing_extensions.TypeVar("T")


class ContextObject(Generic[T]):
    global_v: typing.Optional[T] = None
    ctx: contextvars.ContextVar[T]

    def __init__(self, name):
        self.ctx = contextvars.ContextVar(f"ctx.{name}")

    def set_global(self, v: T):
        self.global_v = v

    def get(self) -> typing.Optional[T]:
        return self.ctx.get(self.global_v)

    @contextlib.contextmanager
    def using(self, v: T):
        token = self.ctx.set(v)
        yield
        self.ctx.reset(token)


class ResourceMutex:
    def __init__(self, v):
        self.__lock = threading.Lock()
        self._v = v

    def __enter__(self) -> "Resource":
        self.__lock.acquire()
        return Resource(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__lock.release()


class Resource:
    def __init__(self, m: ResourceMutex):
        self.m = m

    def get(self):
        return self.m._v

    def set(self, v):
        self.m._v = v
