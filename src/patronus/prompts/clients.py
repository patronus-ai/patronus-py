import abc
import asyncio
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Optional, Type, Union, Literal, NamedTuple, cast

import patronus_api

from patronus.prompts.models import LoadedPrompt
from patronus.prompts.templating import (
    TemplateEngine,
    DefaultTemplateEngines,
    get_template_engine,
)
from patronus.utils import NOT_GIVEN
from patronus import config
from patronus import context


class PromptNotFoundError(Exception):
    """Raised when a prompt could not be found."""

    def __init__(
        self, name: str, project: Optional[str] = None, revision: Optional[int] = None, label: Optional[str] = None
    ):
        self.name = name
        self.project = project
        self.revision = revision
        self.label = label
        message = f"Prompt not found (name={name!r}, project={project!r}, revision={revision!r}, label={label!r})"
        super().__init__(message)


class PromptProviderError(Exception):
    """Base class for prompt provider errors."""


class PromptProviderConnectionError(PromptProviderError):
    """Raised when there's a connectivity issue with the prompt provider."""


class PromptProviderAuthenticationError(PromptProviderError):
    """Raised when there's an authentication issue with the prompt provider."""


class PromptProvider(abc.ABC):
    @abc.abstractmethod
    def get_prompt(
        self, name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine
    ) -> Optional[LoadedPrompt]:
        """Get prompts, returns None if prompt was not found"""

    @abc.abstractmethod
    async def aget_prompt(
        self, name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine
    ) -> Optional[LoadedPrompt]:
        """Get prompts, returns None if prompt was not found"""


class LocalPromptProvider(PromptProvider):
    def get_prompt(
        self, name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine
    ) -> Optional[LoadedPrompt]:
        # TODO implement later
        return None

    async def aget_prompt(
        self, name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine
    ) -> Optional[LoadedPrompt]:
        # TODO implement later
        return None


class APIPromptProvider(PromptProvider):
    def get_prompt(
        self, name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine
    ) -> Optional[LoadedPrompt]:
        cli = context.get_api_client().prompts
        params = self._prepare_params(name, revision, label, project)

        resp = cli.list_revisions(**params)
        if not resp.prompt_revisions:
            return None

        prompt_revision = resp.prompt_revisions[0]
        resp_pd = cli.list_definitions(prompt_id=prompt_revision.prompt_definition_id, limit=1)

        if not resp_pd.prompt_definitions:
            raise PromptProviderError(
                "Prompt revision has been found but prompt definition was not found. This should not happen"
            )

        return self._create_loaded_prompt(prompt_revision, resp_pd.prompt_definitions[0], engine)

    async def aget_prompt(
        self, name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine
    ) -> Optional[LoadedPrompt]:
        cli = context.get_async_api_client().prompts
        params = self._prepare_params(name, revision, label, project)

        resp = await cli.list_revisions(**params)
        if not resp.prompt_revisions:
            return None

        prompt_revision = resp.prompt_revisions[0]
        resp_pd = await cli.list_definitions(prompt_id=prompt_revision.prompt_definition_id, limit=1)

        if not resp_pd.prompt_definitions:
            raise PromptProviderError(
                "Prompt revision has been found but prompt definition was not found. This should not happen"
            )

        return self._create_loaded_prompt(prompt_revision, resp_pd.prompt_definitions[0], engine)

    @staticmethod
    def _prepare_params(name: str, revision: Optional[int], label: Optional[str], project: str) -> dict:
        return {
            "prompt_name": name,
            "revision": revision or patronus_api.NOT_GIVEN,
            "label": label or patronus_api.NOT_GIVEN,
            "project_name": project,
        }

    @staticmethod
    def _create_loaded_prompt(prompt_revision, prompt_def, engine: TemplateEngine) -> LoadedPrompt:
        return LoadedPrompt(
            prompt_definition_id=prompt_revision.id,
            project_id=prompt_revision.project_id,
            project_name=prompt_revision.project_name,
            name=prompt_revision.prompt_definition_name,
            description=prompt_def.description,
            revision_id=prompt_revision.id,
            revision=prompt_revision.revision,
            body=prompt_revision.body,
            normalized_body_sha256=prompt_revision.normalized_body_sha256,
            metadata=prompt_revision.metadata,
            labels=prompt_revision.labels,
            created_at=prompt_revision.created_at,
            _engine=engine,
        )


_DefaultProviders = Literal["local", "api"]
ProviderFactory = dict[str, Callable[[], PromptProvider]]


class _CacheKey(NamedTuple):
    project_name: str
    prompt_name: str
    revision: Optional[int]
    label: Optional[str]


class PromptCache:
    def __init__(self) -> None:
        self._cache: dict[_CacheKey, LoadedPrompt] = {}
        self._mutex = threading.Lock()

    def get(self, key: _CacheKey) -> Optional[LoadedPrompt]:
        with self._mutex:
            return self._cache.get(key)

    def put(self, key: _CacheKey, prompt: LoadedPrompt) -> None:
        with self._mutex:
            self._cache[key] = prompt

    def clear(self) -> None:
        with self._mutex:
            self._cache.clear()


class AsyncPromptCache:
    def __init__(self) -> None:
        self._cache: dict[_CacheKey, LoadedPrompt] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: _CacheKey) -> Optional[LoadedPrompt]:
        async with self._lock:
            return self._cache.get(key)

    async def put(self, key: _CacheKey, prompt: LoadedPrompt) -> None:
        async with self._lock:
            self._cache[key] = prompt

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()


class PromptClientMixin:
    @staticmethod
    def _resolve_project(project: Union[str, Type[NOT_GIVEN]]) -> str:
        """Resolve project name from input or config."""
        if project is not NOT_GIVEN:
            return cast(str, project)

        _scope = context.get_scope_or_none()
        project_name: Optional[str] = None
        if _scope is not None:
            project_name = _scope.project_name
        if project_name is None:
            project_name = config.config().project_name

        return project_name

    @staticmethod
    def _resolve_engine(engine: Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]]) -> TemplateEngine:
        """Resolve template engine from input or config."""
        if engine is NOT_GIVEN:
            engine = context.get_prompts_config().templating_engine

        return get_template_engine(engine)

    @staticmethod
    def _resolve_providers(
        provider: Union[
            PromptProvider, _DefaultProviders, Sequence[Union[PromptProvider, _DefaultProviders]], Type[NOT_GIVEN]
        ],
        provider_factory: Mapping[str, Callable[[], PromptProvider]],
    ) -> list[PromptProvider]:
        """Resolve provider(s) from input or config."""
        if provider is NOT_GIVEN:
            provider = context.get_prompts_config().providers

        if isinstance(provider, (str, PromptProvider)):
            provider = [provider]

        resolved_providers: list[PromptProvider] = []
        for prompt_provider in provider:
            if isinstance(prompt_provider, str) and prompt_provider in provider_factory:
                prompt_provider = provider_factory[prompt_provider]()
            if not isinstance(prompt_provider, PromptProvider):
                raise ValueError("Provided provider must be an instance of PromptProvider")
            resolved_providers.append(prompt_provider)

        return resolved_providers

    @staticmethod
    def _format_provider_errors(provider_errors: list[str]) -> str:
        """Format provider errors for error messages."""
        if not provider_errors:
            return ""
        error_msg = "\n".join([f"  - {err}" for err in provider_errors])
        return f"Provider errors:\n{error_msg}"


class PromptClient(PromptClientMixin):
    def __init__(self, provider_factory: Optional[ProviderFactory] = None) -> None:
        self._cache: PromptCache = PromptCache()
        self._provider_factory: ProviderFactory = provider_factory or {
            "local": lambda: LocalPromptProvider(),
            "api": lambda: APIPromptProvider(),
        }

    def get(
        self,
        name: str,
        revision: Optional[int] = None,
        label: Optional[str] = None,
        project: Union[str, Type[NOT_GIVEN]] = NOT_GIVEN,
        disable_cache: bool = False,
        provider: Union[
            PromptProvider,
            _DefaultProviders,
            Sequence[Union[PromptProvider, _DefaultProviders]],
            Type[NOT_GIVEN],
        ] = NOT_GIVEN,
        engine: Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]] = NOT_GIVEN,
    ) -> LoadedPrompt:
        """
        Get the prompt.
        If neither revision nor label is specified then the prompt with latest revision is returned.

        Project is loaded from the config by default.
        You can specify the project name of the prompt if you want to override the value from the config.

        By default, once a prompt is retrieved it's cached. You can disable caching.

        Args:
            name: The name of the prompt to retrieve.
            revision: Optional specific revision number to retrieve. If not specified, the latest revision is used.
            label: Optional label to filter by. If specified, only prompts with this label will be returned.
            project: Optional project name override. If not specified, the project name from config is used.
            disable_cache: If True, bypasses the cache for both reading and writing.
            provider: The provider(s) to use for retrieving prompts. Can be a string identifier ('local', 'api'),
                     a PromptProvider instance, or a sequence of these. If not specified, defaults to config setting.
            engine: The template engine to use for rendering prompts. Can be a string identifier ('f-string', 'mustache', 'jinja2')
                   or a TemplateEngine instance. If not specified, defaults to config setting.

        Returns:
            LoadedPrompt: The retrieved prompt object.

        Raises:
            PromptNotFoundError: If the prompt could not be found with the specified parameters.
            ValueError: If the provided provider or engine is invalid.
            PromptProviderError: If there was an error communicating with the prompt provider.
        """
        project_name: str = self._resolve_project(project)
        resolved_providers: list[PromptProvider] = self._resolve_providers(provider, self._provider_factory)
        resolved_engine: TemplateEngine = self._resolve_engine(engine)

        cache_key: _CacheKey = _CacheKey(project_name=project_name, prompt_name=name, revision=revision, label=label)
        if not disable_cache:
            cached_prompt: Optional[LoadedPrompt] = self._cache.get(cache_key)
            if cached_prompt is not None:
                return cached_prompt

        prompt: Optional[LoadedPrompt] = None
        provider_errors: list[str] = []

        for prompt_provider in resolved_providers:
            try:
                prompt = prompt_provider.get_prompt(name, revision, label, project_name, engine=resolved_engine)
                if prompt is not None:
                    break
            except PromptProviderConnectionError as e:
                provider_errors.append(str(e))
                continue
            except PromptProviderAuthenticationError as e:
                provider_errors.append(str(e))
                continue
            except Exception as e:
                provider_errors.append(f"Unexpected error from provider {prompt_provider.__class__.__name__}: {str(e)}")
                continue

        if prompt is None:
            if provider_errors:
                error_msg: str = self._format_provider_errors(provider_errors)
                raise PromptNotFoundError(
                    name=name, project=project_name, revision=revision, label=label
                ) from Exception(error_msg)
            else:
                raise PromptNotFoundError(name=name, project=project_name, revision=revision, label=label)

        if not disable_cache:
            self._cache.put(cache_key, prompt)

        return prompt


class AsyncPromptClient(PromptClientMixin):
    def __init__(self, provider_factory: Optional[ProviderFactory] = None) -> None:
        self._cache: AsyncPromptCache = AsyncPromptCache()
        self._provider_factory: ProviderFactory = provider_factory or {
            "local": lambda: LocalPromptProvider(),
            "api": lambda: APIPromptProvider(),
        }

    async def get(
        self,
        name: str,
        revision: Optional[int] = None,
        label: Optional[str] = None,
        project: Union[str, Type[NOT_GIVEN]] = NOT_GIVEN,
        disable_cache: bool = False,
        provider: Union[
            PromptProvider, _DefaultProviders, Sequence[Union[PromptProvider, _DefaultProviders]], Type[NOT_GIVEN]
        ] = NOT_GIVEN,
        engine: Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]] = NOT_GIVEN,
    ) -> LoadedPrompt:
        """
        Get the prompt asynchronously.
        If neither revision nor label is specified then the prompt with latest revision is returned.

        Project is loaded from the config by default.
        You can specify the project name of the prompt if you want to override the value from the config.

        By default, once a prompt is retrieved it's cached. You can disable caching.

        Args:
            name: The name of the prompt to retrieve.
            revision: Optional specific revision number to retrieve. If not specified, the latest revision is used.
            label: Optional label to filter by. If specified, only prompts with this label will be returned.
            project: Optional project name override. If not specified, the project name from config is used.
            disable_cache: If True, bypasses the cache for both reading and writing.
            provider: The provider(s) to use for retrieving prompts. Can be a string identifier ('local', 'api'),
                     a PromptProvider instance, or a sequence of these. If not specified, defaults to config setting.
            engine: The template engine to use for rendering prompts. Can be a string identifier ('f-string', 'mustache', 'jinja2')
                   or a TemplateEngine instance. If not specified, defaults to config setting.

        Returns:
            LoadedPrompt: The retrieved prompt object.

        Raises:
            PromptNotFoundError: If the prompt could not be found with the specified parameters.
            ValueError: If the provided provider or engine is invalid.
            PromptProviderError: If there was an error communicating with the prompt provider.
        """
        project_name: str = self._resolve_project(project)
        resolved_providers: list[PromptProvider] = self._resolve_providers(provider, self._provider_factory)
        resolved_engine: TemplateEngine = self._resolve_engine(engine)

        cache_key: _CacheKey = _CacheKey(project_name=project_name, prompt_name=name, revision=revision, label=label)
        if not disable_cache:
            cached_prompt: Optional[LoadedPrompt] = await self._cache.get(cache_key)
            if cached_prompt is not None:
                return cached_prompt

        prompt: Optional[LoadedPrompt] = None
        provider_errors: list[str] = []

        for prompt_provider in resolved_providers:
            try:
                prompt = await prompt_provider.aget_prompt(name, revision, label, project_name, engine=resolved_engine)
                if prompt is not None:
                    break
            except PromptProviderConnectionError as e:
                provider_errors.append(str(e))
                continue
            except PromptProviderAuthenticationError as e:
                provider_errors.append(str(e))
                continue
            except Exception as e:
                provider_errors.append(f"Unexpected error from provider {prompt_provider.__class__.__name__}: {str(e)}")
                continue

        if prompt is None:
            if provider_errors:
                error_msg: str = self._format_provider_errors(provider_errors)
                raise PromptNotFoundError(
                    name=name, project=project_name, revision=revision, label=label
                ) from Exception(error_msg)
            else:
                raise PromptNotFoundError(name=name, project=project_name, revision=revision, label=label)

        if not disable_cache:
            await self._cache.put(cache_key, prompt)

        return prompt


_default_client: PromptClient = PromptClient()
_default_async_client: AsyncPromptClient = AsyncPromptClient()

load_prompt = _default_client.get
aload_prompt = _default_async_client.get
