import abc
import asyncio
import json
import logging
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Optional, Type, Union, Literal, NamedTuple, cast

import patronus_api

from patronus import config
from patronus import context
from patronus.prompts.models import LoadedPrompt, Prompt, calculate_normalized_body_hash
from patronus.prompts.templating import (
    TemplateEngine,
    DefaultTemplateEngines,
    get_template_engine,
)
from patronus.utils import NOT_GIVEN

log = logging.getLogger("patronus.core")


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
            result = self._cache.get(key)
            if result is not None:
                log.debug(
                    "Cache hit for prompt %s (project=%s, revision=%s, label=%s)",
                    key.prompt_name,
                    key.project_name,
                    key.revision,
                    key.label,
                )
            else:
                log.debug(
                    "Cache miss for prompt %s (project=%s, revision=%s, label=%s)",
                    key.prompt_name,
                    key.project_name,
                    key.revision,
                    key.label,
                )
            return result

    def put(self, key: _CacheKey, prompt: LoadedPrompt) -> None:
        with self._mutex:
            self._cache[key] = prompt
            log.debug(
                "Cached prompt %s (project=%s, revision=%d, labels=%s)",
                prompt.name,
                prompt.project_name,
                prompt.revision,
                ",".join(prompt.labels) if prompt.labels else "none",
            )

    def clear(self) -> None:
        with self._mutex:
            log.debug("Clearing prompt cache with %d entries", len(self._cache))
            self._cache.clear()


class AsyncPromptCache:
    def __init__(self) -> None:
        self._cache: dict[_CacheKey, LoadedPrompt] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: _CacheKey) -> Optional[LoadedPrompt]:
        async with self._lock:
            result = self._cache.get(key)
            if result is not None:
                log.debug(
                    "Async cache hit for prompt %s (project=%s, revision=%s, label=%s)",
                    key.prompt_name,
                    key.project_name,
                    key.revision,
                    key.label,
                )
            else:
                log.debug(
                    "Async cache miss for prompt %s (project=%s, revision=%s, label=%s)",
                    key.prompt_name,
                    key.project_name,
                    key.revision,
                    key.label,
                )
            return result

    async def put(self, key: _CacheKey, prompt: LoadedPrompt) -> None:
        async with self._lock:
            self._cache[key] = prompt
            log.debug(
                "Cached prompt in async cache %s (project=%s, revision=%d, labels=%s)",
                prompt.name,
                prompt.project_name,
                prompt.revision,
                ",".join(prompt.labels) if prompt.labels else "none",
            )

    async def clear(self) -> None:
        async with self._lock:
            log.debug("Clearing async prompt cache with %d entries", len(self._cache))
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
        self._api_provider = APIPromptProvider()

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

        for i, prompt_provider in enumerate(resolved_providers):
            log.debug("Trying prompt provider %d (%s)", i + 1, prompt_provider.__class__.__name__)
            try:
                prompt = prompt_provider.get_prompt(name, revision, label, project_name, engine=resolved_engine)
                if prompt is not None:
                    log.debug("Prompt found using provider %s", prompt_provider.__class__.__name__)
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

    def push(
        self,
        prompt: Prompt,
        project: Union[str, Type[NOT_GIVEN]] = NOT_GIVEN,
        engine: Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]] = NOT_GIVEN,
    ) -> LoadedPrompt:
        """
        Push a prompt to the API, creating a new revision only if needed.

        If a prompt revision with the same normalized body and metadata already exists,
        the existing revision will be returned. If the metadata differs, a new revision will be created.

        The engine parameter is only used to set property on output LoadedPrompt object.
        It is not persisted in any way and doesn't affect how the prompt is stored in Patronus AI Platform.

        Note that when a new prompt definition is created, the description is used as provided.
        However, when creating a new revision for an existing prompt definition, the
        description parameter doesn't update the existing prompt definition's description.

        Args:
            prompt: The prompt to push
            project: Optional project name override. If not specified, the project name from config is used.
            engine: The template engine to use for rendering the returned prompt. If not specified, defaults to config setting.

        Returns:
            LoadedPrompt: The created or existing prompt revision

        Raises:
            PromptProviderError: If there was an error communicating with the prompt provider.
        """
        project_name: str = self._resolve_project(project)
        resolved_engine: TemplateEngine = self._resolve_engine(engine)

        normalized_body_sha256 = calculate_normalized_body_hash(prompt.body)

        cli = context.get_api_client().prompts
        # Try to find existing revision with same hash
        resp = cli.list_revisions(
            prompt_name=prompt.name,
            project_name=project_name,
            normalized_body_sha256=normalized_body_sha256,
        )

        # Variables for create_revision parameters
        prompt_id = patronus_api.NOT_GIVEN
        prompt_name = prompt.name
        create_new_prompt = True
        prompt_def = None

        # If we found a matching revision, check if metadata is the same
        if resp.prompt_revisions:
            log.debug("Found %d revisions with matching body hash", len(resp.prompt_revisions))
            prompt_id = resp.prompt_revisions[0].prompt_definition_id
            create_new_prompt = False

            resp_pd = cli.list_definitions(prompt_id=prompt_id, limit=1)
            if not resp_pd.prompt_definitions:
                raise PromptProviderError(
                    "Prompt revision has been found but prompt definition was not found. This should not happen"
                )
            prompt_def = resp_pd.prompt_definitions[0]

            # Check if the provided description is different from existing one and warn if so
            if prompt.description is not None and prompt.description != prompt_def.description:
                warnings.warn(
                    f"Prompt description ({prompt.description!r}) differs from the existing one "
                    f"({prompt_def.description!r}). The description won't be updated."
                )

            new_metadata_cmp = json.dumps(prompt.metadata, sort_keys=True)
            for rev in resp.prompt_revisions:
                metadata_cmp = json.dumps(rev.metadata, sort_keys=True)
                if new_metadata_cmp == metadata_cmp:
                    log.debug("Found existing revision with matching metadata, returning revision %d", rev.revision)
                    return self._api_provider._create_loaded_prompt(
                        prompt_revision=rev,
                        prompt_def=prompt_def,
                        engine=resolved_engine,
                    )

            # For existing prompt, don't need name/project
            prompt_name = patronus_api.NOT_GIVEN
            project_name = patronus_api.NOT_GIVEN
        else:
            # No matching revisions found, will create new prompt
            log.debug("No revisions with matching body hash found, creating new prompt and revision")

        # Create a new revision with appropriate parameters
        log.debug(
            "Creating new revision (new_prompt=%s, prompt_id=%s, prompt_name=%s)",
            create_new_prompt,
            prompt_id if prompt_id != patronus_api.NOT_GIVEN else "NOT_GIVEN",
            prompt_name if prompt_name != patronus_api.NOT_GIVEN else "NOT_GIVEN",
        )
        resp = cli.create_revision(
            body=prompt.body,
            prompt_id=prompt_id,
            prompt_name=prompt_name,
            project_name=project_name if create_new_prompt else patronus_api.NOT_GIVEN,
            prompt_description=prompt.description,
            metadata=prompt.metadata,
        )

        prompt_revision = resp.prompt_revision

        # If we created a new prompt, we need to fetch the definition
        if create_new_prompt:
            resp_pd = cli.list_definitions(prompt_id=prompt_revision.prompt_definition_id, limit=1)
            if not resp_pd.prompt_definitions:
                raise PromptProviderError(
                    "Prompt revision has been created but prompt definition was not found. This should not happen"
                )
            prompt_def = resp_pd.prompt_definitions[0]

        return self._api_provider._create_loaded_prompt(prompt_revision, prompt_def, resolved_engine)


class AsyncPromptClient(PromptClientMixin):
    def __init__(self, provider_factory: Optional[ProviderFactory] = None) -> None:
        self._cache: AsyncPromptCache = AsyncPromptCache()
        self._provider_factory: ProviderFactory = provider_factory or {
            "local": lambda: LocalPromptProvider(),
            "api": lambda: APIPromptProvider(),
        }
        self._api_provider = APIPromptProvider()

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

        for i, prompt_provider in enumerate(resolved_providers):
            log.debug("Trying prompt provider %d (%s) async", i + 1, prompt_provider.__class__.__name__)
            try:
                prompt = await prompt_provider.aget_prompt(name, revision, label, project_name, engine=resolved_engine)
                if prompt is not None:
                    log.debug("Prompt found using async provider %s", prompt_provider.__class__.__name__)
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

    async def push(
        self,
        prompt: Prompt,
        project: Union[str, Type[NOT_GIVEN]] = NOT_GIVEN,
        engine: Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]] = NOT_GIVEN,
    ) -> LoadedPrompt:
        """
        Push a prompt to the API asynchronously, creating a new revision only if needed.

        If a prompt revision with the same normalized body and metadata already exists,
        the existing revision will be returned. If the metadata differs, a new revision will be created.

        The engine parameter is only used to set property on output LoadedPrompt object.
        It is not persisted in any way and doesn't affect how the prompt is stored in Patronus AI Platform.

        Note that when a new prompt definition is created, the description is used as provided.
        However, when creating a new revision for an existing prompt definition, the
        description parameter doesn't update the existing prompt definition's description.

        Args:
            prompt: The prompt to push
            project: Optional project name override. If not specified, the project name from config is used.
            engine: The template engine to use for rendering the returned prompt. If not specified, defaults to config setting.

        Returns:
            LoadedPrompt: The created or existing prompt revision

        Raises:
            PromptProviderError: If there was an error communicating with the prompt provider.
        """
        project_name: str = self._resolve_project(project)
        resolved_engine: TemplateEngine = self._resolve_engine(engine)

        normalized_body_sha256 = calculate_normalized_body_hash(prompt.body)

        cli = context.get_async_api_client().prompts
        # Try to find existing revision with same hash
        resp = await cli.list_revisions(
            prompt_name=prompt.name,
            project_name=project_name,
            normalized_body_sha256=normalized_body_sha256,
        )

        # Variables for create_revision parameters
        prompt_id = patronus_api.NOT_GIVEN
        prompt_name = prompt.name
        create_new_prompt = True
        prompt_def = None

        # If we found a matching revision, check if metadata is the same
        if resp.prompt_revisions:
            log.debug("Found %d revisions with matching body hash", len(resp.prompt_revisions))
            prompt_id = resp.prompt_revisions[0].prompt_definition_id
            create_new_prompt = False

            resp_pd = await cli.list_definitions(prompt_id=prompt_id, limit=1)
            if not resp_pd.prompt_definitions:
                raise PromptProviderError(
                    "Prompt revision has been found but prompt definition was not found. This should not happen"
                )
            prompt_def = resp_pd.prompt_definitions[0]

            # Check if the provided description is different from existing one and warn if so
            if prompt.description is not None and prompt.description != prompt_def.description:
                warnings.warn(
                    f"Prompt description ({prompt.description!r}) differs from the existing one "
                    f"({prompt_def.description!r}). The description won't be updated."
                )

            new_metadata_cmp = json.dumps(prompt.metadata, sort_keys=True)
            for rev in resp.prompt_revisions:
                metadata_cmp = json.dumps(rev.metadata, sort_keys=True)
                if new_metadata_cmp == metadata_cmp:
                    log.debug("Found existing revision with matching metadata, returning revision %d", rev.revision)
                    return self._api_provider._create_loaded_prompt(
                        prompt_revision=rev,
                        prompt_def=prompt_def,
                        engine=resolved_engine,
                    )

            # For existing prompt, don't need name/project
            prompt_name = patronus_api.NOT_GIVEN
            project_name = patronus_api.NOT_GIVEN
        else:
            # No matching revisions found, will create new prompt
            log.debug("No revisions with matching body hash found, creating new prompt and revision")

        # Create a new revision with appropriate parameters
        log.debug(
            "Creating new revision (new_prompt=%s, prompt_id=%s, prompt_name=%s)",
            create_new_prompt,
            prompt_id if prompt_id != patronus_api.NOT_GIVEN else "NOT_GIVEN",
            prompt_name if prompt_name != patronus_api.NOT_GIVEN else "NOT_GIVEN",
        )
        resp = await cli.create_revision(
            body=prompt.body,
            prompt_id=prompt_id,
            prompt_name=prompt_name,
            project_name=project_name if create_new_prompt else patronus_api.NOT_GIVEN,
            prompt_description=prompt.description,
            metadata=prompt.metadata,
        )

        prompt_revision = resp.prompt_revision

        # If we created a new prompt, we need to fetch the definition
        if create_new_prompt:
            resp_pd = await cli.list_definitions(prompt_id=prompt_revision.prompt_definition_id, limit=1)
            if not resp_pd.prompt_definitions:
                raise PromptProviderError(
                    "Prompt revision has been created but prompt definition was not found. This should not happen"
                )
            prompt_def = resp_pd.prompt_definitions[0]

        return self._api_provider._create_loaded_prompt(prompt_revision, prompt_def, resolved_engine)


_default_client: PromptClient = PromptClient()
_default_async_client: AsyncPromptClient = AsyncPromptClient()

load_prompt = _default_client.get
"""Alias for [PromptClient.get][patronus.prompts.clients.PromptClient.get]."""

aload_prompt = _default_async_client.get
"""Alias for [AsyncPromptClient.get][patronus.prompts.clients.AsyncPromptClient.get]."""

push_prompt = _default_client.push
"""Alias for [PromptClient.push][patronus.prompts.clients.PromptClient.push]."""

apush_prompt = _default_async_client.push
"""Alias for [AsyncPromptClient.push][patronus.prompts.clients.AsyncPromptClient.push]."""
