import abc
from collections.abc import Sequence
from typing import Type, Union, Literal, Optional, NamedTuple

import patronus_api

from patronus.prompts.models import LoadedPrompt
from patronus.prompts.templating import (
    TemplateEngine,
    FStringTemplateEngine,
    MustacheTemplateEngine,
    Jinja2TemplateEngine,
)
from patronus.utils import NOT_GIVEN
from patronus import config
from patronus import context


class PromptNotFoundError(ValueError):
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

    pass


class PromptProviderConnectionError(PromptProviderError):
    """Raised when there's a connectivity issue with the prompt provider."""

    pass


class PromptProviderAuthenticationError(PromptProviderError):
    """Raised when there's an authentication issue with the prompt provider."""

    pass


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
        resp = cli.list_revisions(
            prompt_name=name,
            revision=revision or patronus_api.NOT_GIVEN,
            label=label or patronus_api.NOT_GIVEN,
            project_name=project,
        )
        if not resp.prompt_revisions:
            return None
        revision = resp.prompt_revisions[0]
        resp_pd = cli.list_definitions(prompt_id=revision.prompt_definition_id, limit=1)
        if not resp_pd.prompt_definitions:
            raise PromptProviderError(
                "Prompt revision has been found but prompt definition was not found. This should not happen"
            )
        prompt_def = resp_pd.prompt_definitions[0]
        return LoadedPrompt(
            prompt_definition_id=revision.id,
            project_id=revision.project_id,
            project_name=revision.project_name,
            name=revision.prompt_definition_name,
            description=prompt_def.description,
            revision_id=revision.id,
            revision=revision.revision,
            body=revision.body,
            normalized_body_sha256=revision.normalized_body_sha256,
            metadata=revision.metadata,
            labels=revision.labels,
            created_at=revision.created_at,
            _engine=engine,
        )

    async def aget_prompt(
        self, name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine
    ) -> Optional[LoadedPrompt]:
        cli = context.get_async_api_client().prompts
        resp = await cli.list_revisions(
            prompt_name=name,
            revision=revision or patronus_api.NOT_GIVEN,
            label=label or patronus_api.NOT_GIVEN,
            project_name=project,
        )
        if not resp.prompt_revisions:
            return None
        revision = resp.prompt_revisions[0]
        resp_pd = await cli.list_definitions(prompt_id=revision.prompt_definition_id, limit=1)
        if not resp_pd.prompt_definitions:
            raise PromptProviderError(
                "Prompt revision has been found but prompt definition was not found. This should not happen"
            )
        prompt_def = resp_pd.prompt_definitions[0]
        return LoadedPrompt(
            prompt_definition_id=revision.id,
            project_id=revision.project_id,
            project_name=revision.project_name,
            name=revision.prompt_definition_name,
            description=prompt_def.description,
            revision_id=revision.id,
            revision=revision.revision,
            body=revision.body,
            normalized_body_sha256=revision.normalized_body_sha256,
            metadata=revision.metadata,
            labels=revision.labels,
            created_at=revision.created_at,
            _engine=engine,
        )


_DefaultProviders = Literal["local", "api"]
_DefaultTemplateEngines = Literal["f-string", "mustache", "jinja2"]


class _CacheKey(NamedTuple):
    project_name: str
    prompt_name: str
    revision: Optional[int]
    label: Optional[str]


class PromptCache:
    """Thread-safe cache for prompt objects."""

    def __init__(self):
        self._cache = {}
        self._mutex = __import__("threading").Lock()

    def get(self, key: _CacheKey) -> Optional[LoadedPrompt]:
        """Get a prompt from the cache."""
        with self._mutex:
            return self._cache.get(key)

    def put(self, key: _CacheKey, prompt: LoadedPrompt) -> None:
        """Store a prompt in the cache."""
        with self._mutex:
            self._cache[key] = prompt

    def clear(self) -> None:
        """Clear the cache."""
        with self._mutex:
            self._cache.clear()


# TODO need async version (later)
class PromptClient:
    def __init__(self):
        self._cache = PromptCache()

    def get(
        self,
        name: str,
        revision=None,
        label=None,
        project: Union[str, Type[NOT_GIVEN]] = NOT_GIVEN,
        disable_cache: bool = False,
        provider: Union[
            PromptProvider, _DefaultProviders, Sequence[Union[PromptProvider, _DefaultProviders]], Type[NOT_GIVEN]
        ] = NOT_GIVEN,
        engine: Union[TemplateEngine, _DefaultProviders, Type[NOT_GIVEN]] = NOT_GIVEN,
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
        # Resolve project from config if needed
        project_name = None
        if project is not NOT_GIVEN:
            project_name = project
        else:
            _scope = context.get_scope_or_none()
            if _scope is not None:
                project_name = _scope.project_name
            if project_name is None:
                project_name = config.config().project_name

        # Check cache first if not disabled
        cache_key = _CacheKey(project_name=project_name, prompt_name=name, revision=revision, label=label)
        if not disable_cache:
            cached_prompt = self._cache.get(cache_key)
            if cached_prompt is not None:
                return cached_prompt

        # Resolve provider
        if provider is NOT_GIVEN:
            provider = context.get_prompts_config().providers

        # Resolve template engine
        if engine is NOT_GIVEN:
            engine = context.get_prompts_config().templating_engine

        if isinstance(provider, (str, PromptProvider)):
            provider = [provider]

        resolved_providers: list[PromptProvider] = []
        for prompt_provider in provider:
            if prompt_provider == "local":
                prompt_provider = LocalPromptProvider()
            if prompt_provider == "api":
                prompt_provider = APIPromptProvider()
            if not isinstance(prompt_provider, PromptProvider):
                raise ValueError("Provided provider must be an instance of PromptProvider")
            resolved_providers.append(prompt_provider)

        if engine == "f-string":
            engine = FStringTemplateEngine()
        elif engine == "mustache":
            engine = MustacheTemplateEngine()
        elif engine == "jinja2":
            engine = Jinja2TemplateEngine()

        if not isinstance(engine, TemplateEngine):
            raise ValueError(
                "Provided engine must be an instance of TemplateEngine or "
                "one of the default engines ('f-string', 'mustache', 'jinja2'). "
                f"Instead got {engine!r}"
            )

        prompt = None
        provider_errors = []
        for prompt_provider in resolved_providers:
            try:
                prompt = prompt_provider.get_prompt(name, revision, label, project_name, engine=engine)
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
                error_msg = "\n".join([f"  - {err}" for err in provider_errors])
                raise PromptNotFoundError(
                    name=name, project=project_name, revision=revision, label=label
                ) from Exception(f"Provider errors:\n{error_msg}")
            else:
                raise PromptNotFoundError(name=name, project=project_name, revision=revision, label=label)

        if not disable_cache:
            self._cache.put(cache_key, prompt)

        return prompt


_default_client = PromptClient()

load_prompt = _default_client.get
