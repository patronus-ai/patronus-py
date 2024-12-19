import dataclasses
import logging
import platform
import typing
from functools import lru_cache
from typing import Optional

import httpx
import pydantic

log = logging.getLogger(__name__)


class APIError(Exception):
    response: httpx.Response

    def __init__(self, message: Optional[str] = None, response: Optional[httpx.Response] = None):
        self.response = response
        super().__init__(message)


class ClientError(APIError):
    def __init__(self, response: httpx.Response):
        self.response = response
        super().__init__(f"Client error '{response.status_code}' for url '{response.request.url}: {response.text}'")


class UnrecoverableAPIError(APIError):
    def __init__(self, message: str, response: httpx.Response):
        message = f"{self.__class__.__name__}: {message}: response content: {response.text}"
        super().__init__(message, response)


class RPMLimitError(APIError):
    def __init__(self, limit: int, wait_for_s: Optional[float], response: httpx.Response):
        self.limit = limit
        self.wait_for_s = wait_for_s
        super().__init__(
            f"RPMLimitError: Request per minute limit hit (quota limit: {self.limit})",
            response=response,
        )


class RetryError(Exception):
    stack_trace: str

    def __init__(self, attempt: int, out_of: int, origin: Exception, stack_trace: str):
        self.stack_trace = stack_trace
        super().__init__(f"RetryError: execution failed after {attempt}/{out_of} attempts: {origin}")


R = typing.TypeVar("R", bound=pydantic.BaseModel)


@dataclasses.dataclass
class CallResponse(typing.Generic[R]):
    data: Optional[R]
    response: httpx.Response

    def raise_for_status(self):
        if self.response.status_code in (400, 422):
            raise ClientError(self.response)
        self.response.raise_for_status()


class BaseAPIClient:
    version: str
    http: httpx.AsyncClient
    http_sync: httpx.Client
    base_url: str = None
    api_key: str = None

    def __init__(self, *, version: str, http: httpx.AsyncClient, http_sync: httpx.Client):
        self.version = version
        self.http = http
        self.http_sync = http_sync

    def set_target(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def call(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, typing.Any]] = None,
        body: Optional[pydantic.BaseModel] = None,
        response_cls: Optional[type[R]],
    ) -> CallResponse[R]:
        content = None
        if body is not None:
            content = body.model_dump_json(by_alias=True)

        url = self._url(path)
        log.debug(f"Sending HTTP request {url!r}")
        response = await self.http.request(
            method,
            url,
            params=params,
            content=content,
            headers=self.headers(),
        )
        return self._call_process_resp(url, response, response_cls)

    def call_sync(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, typing.Any]] = None,
        body: Optional[pydantic.BaseModel] = None,
        response_cls: Optional[type[R]],
    ) -> CallResponse[R]:
        content = None
        if body is not None:
            content = body.model_dump_json(by_alias=True)

        url = self._url(path)
        log.debug(f"Sending HTTP request {url!r}")
        response = self.http_sync.request(
            method,
            url,
            params=params,
            content=content,
            headers=self.headers(),
        )
        return self._call_process_resp(url, response, response_cls)

    @staticmethod
    def _call_process_resp(url: str, response: httpx.Response, response_cls: Optional[type[R]]) -> CallResponse[R]:
        log.debug(f"Received HTTP response {url!r} {response.status_code}")

        data = None
        if response.is_error:
            log.debug(f"Response with error: content: {response.text}")
        if response.is_success and response_cls is not None:
            data = response_cls.model_validate_json(response.content)

        return CallResponse(
            data=data,
            response=response,
        )

    def _url(self, path: str):
        assert self.base_url, "BaseAPIClient: base_url must be set"
        return f"{self.base_url}{path}"

    def headers(self) -> typing.Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
            **self.auth_headers(),
            **self.platform_headers(),
        }

    @property
    def user_agent(self) -> str:
        return f"PatronusAI/Python {self.version}"

    def auth_headers(self) -> typing.Dict[str, str]:
        assert self.api_key, "BaseAPIClient.auth_headers(): api_key must be set"
        return {
            "x-api-key": self.api_key,
        }

    @lru_cache(maxsize=None)
    def platform_headers(self) -> typing.Dict[str, str]:
        return {
            "X-Package-Version": self.version,
            "X-Platform-OS": get_platform_os(),
            "X-Platform-Arch": get_platform_arch(),
            "X-Runtime": get_python_runtime(),
            "X-Runtime-version": get_python_version(),
        }


# get_platform_os is based on implementation found in Open AI SDK
# https://github.com/openai/openai-python/blob/631a2a7156299351874f37c4769308a104ce19ed/src/openai/_base_client.py#L1933-L1972
def get_platform_os() -> str:
    try:
        system = platform.system().lower()
        platform_name = platform.platform().lower()
    except Exception:
        return "Unknown"

    if "iphone" in platform_name or "ipad" in platform_name:
        return "iOS"

    if system == "darwin":
        return "MacOS"

    if system == "windows":
        return "Windows"

    if "android" in platform_name:
        return "Android"

    if system == "linux":
        return "Linux"

    if platform_name:
        return f"Other:{platform_name}"

    return "Unknown"


def get_platform_arch():
    try:
        python_bitness, _ = platform.architecture()
        machine = platform.machine().lower()
    except Exception:
        return "Unknown"

    if machine in ("arm64", "aarch64"):
        return "arm64"

    if machine == "arm":
        return "arm"

    if machine == "x86_64":
        return "x64"

    if python_bitness == "32bit":
        return "x32"

    return "Unknown"


def get_python_runtime() -> str:
    try:
        return platform.python_implementation()
    except Exception:
        return "Unknown"


def get_python_version() -> str:
    try:
        return platform.python_version()
    except Exception:
        return "Unknown"
