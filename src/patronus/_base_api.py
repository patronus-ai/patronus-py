import dataclasses
import logging
import platform
import typing
from functools import lru_cache

import httpx
import pydantic

R = typing.TypeVar("R", bound=pydantic.BaseModel)

log = logging.getLogger(__name__)


@dataclasses.dataclass
class CallResponse(typing.Generic[R]):
    data: R | None
    response: httpx.Response


class BaseAPIClient:
    base_url: str = None
    api_key: str = None

    def __init__(self, *, version: str, http: httpx.AsyncClient):
        self.version = version
        self.http = http

    def set_target(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def call(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, typing.Any] | None = None,
        body: pydantic.BaseModel | None = None,
        response_cls: type[R] | None,
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
        log.debug(f"Received HTTP response {url!r} {response.status_code}")

        if response.is_error:
            log.error(f"{url!r} {response.status_code}: {response.text}")
        response.raise_for_status()

        data = None
        if response_cls is not None:
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
