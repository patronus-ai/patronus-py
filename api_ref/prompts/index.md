# Prompts

## patronus.prompts

### clients

#### load_prompt

```python
load_prompt = get
```

Alias for PromptClient.get.

#### aload_prompt

```python
aload_prompt = get
```

Alias for AsyncPromptClient.get.

#### push_prompt

```python
push_prompt = push
```

Alias for PromptClient.push.

#### apush_prompt

```python
apush_prompt = push
```

Alias for AsyncPromptClient.push.

#### PromptNotFoundError

```python
PromptNotFoundError(name: str, project: Optional[str] = None, revision: Optional[int] = None, label: Optional[str] = None)
```

Bases: `Exception`

Raised when a prompt could not be found.

Source code in `src/patronus/prompts/clients.py`

```python
def __init__(
    self, name: str, project: Optional[str] = None, revision: Optional[int] = None, label: Optional[str] = None
):
    self.name = name
    self.project = project
    self.revision = revision
    self.label = label
    message = f"Prompt not found (name={name!r}, project={project!r}, revision={revision!r}, label={label!r})"
    super().__init__(message)
```

#### PromptProviderError

Bases: `Exception`

Base class for prompt provider errors.

#### PromptProviderConnectionError

Bases: `PromptProviderError`

Raised when there's a connectivity issue with the prompt provider.

#### PromptProviderAuthenticationError

Bases: `PromptProviderError`

Raised when there's an authentication issue with the prompt provider.

#### PromptProvider

Bases: `ABC`

##### get_prompt

```python
get_prompt(name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine) -> Optional[LoadedPrompt]
```

Get prompts, returns None if prompt was not found

Source code in `src/patronus/prompts/clients.py`

```python
@abc.abstractmethod
def get_prompt(
    self, name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine
) -> Optional[LoadedPrompt]:
    """Get prompts, returns None if prompt was not found"""
```

##### aget_prompt

```python
aget_prompt(name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine) -> Optional[LoadedPrompt]
```

Get prompts, returns None if prompt was not found

Source code in `src/patronus/prompts/clients.py`

```python
@abc.abstractmethod
async def aget_prompt(
    self, name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine
) -> Optional[LoadedPrompt]:
    """Get prompts, returns None if prompt was not found"""
```

#### PromptClientMixin

#### PromptClient

```python
PromptClient(provider_factory: Optional[ProviderFactory] = None)
```

Bases: `PromptClientMixin`

Source code in `src/patronus/prompts/clients.py`

```python
def __init__(self, provider_factory: Optional[ProviderFactory] = None) -> None:
    self._cache: PromptCache = PromptCache()
    self._provider_factory: ProviderFactory = provider_factory or {
        "local": lambda: LocalPromptProvider(),
        "api": lambda: APIPromptProvider(),
    }
    self._api_provider = APIPromptProvider()
```

##### get

```python
get(name: str, revision: Optional[int] = None, label: Optional[str] = None, project: Union[str, Type[NOT_GIVEN]] = NOT_GIVEN, disable_cache: bool = False, provider: Union[PromptProvider, _DefaultProviders, Sequence[Union[PromptProvider, _DefaultProviders]], Type[NOT_GIVEN]] = NOT_GIVEN, engine: Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]] = NOT_GIVEN) -> LoadedPrompt
```

Get the prompt. If neither revision nor label is specified then the prompt with latest revision is returned.

Project is loaded from the config by default. You can specify the project name of the prompt if you want to override the value from the config.

By default, once a prompt is retrieved it's cached. You can disable caching.

Parameters:

| Name            | Type                                                                                                            | Description                                                                                                                                                                                  | Default     |
| --------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `name`          | `str`                                                                                                           | The name of the prompt to retrieve.                                                                                                                                                          | *required*  |
| `revision`      | `Optional[int]`                                                                                                 | Optional specific revision number to retrieve. If not specified, the latest revision is used.                                                                                                | `None`      |
| `label`         | `Optional[str]`                                                                                                 | Optional label to filter by. If specified, only prompts with this label will be returned.                                                                                                    | `None`      |
| `project`       | `Union[str, Type[NOT_GIVEN]]`                                                                                   | Optional project name override. If not specified, the project name from config is used.                                                                                                      | `NOT_GIVEN` |
| `disable_cache` | `bool`                                                                                                          | If True, bypasses the cache for both reading and writing.                                                                                                                                    | `False`     |
| `provider`      | `Union[PromptProvider, _DefaultProviders, Sequence[Union[PromptProvider, _DefaultProviders]], Type[NOT_GIVEN]]` | The provider(s) to use for retrieving prompts. Can be a string identifier ('local', 'api'), a PromptProvider instance, or a sequence of these. If not specified, defaults to config setting. | `NOT_GIVEN` |
| `engine`        | `Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]]`                                                | The template engine to use for rendering prompts. Can be a string identifier ('f-string', 'mustache', 'jinja2') or a TemplateEngine instance. If not specified, defaults to config setting.  | `NOT_GIVEN` |

Returns:

| Name           | Type           | Description                  |
| -------------- | -------------- | ---------------------------- |
| `LoadedPrompt` | `LoadedPrompt` | The retrieved prompt object. |

Raises:

| Type                  | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| `PromptNotFoundError` | If the prompt could not be found with the specified parameters. |
| `ValueError`          | If the provided provider or engine is invalid.                  |
| `PromptProviderError` | If there was an error communicating with the prompt provider.   |

Source code in `src/patronus/prompts/clients.py`

```python
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
```

##### push

```python
push(prompt: Prompt, project: Union[str, Type[NOT_GIVEN]] = NOT_GIVEN, engine: Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]] = NOT_GIVEN) -> LoadedPrompt
```

Push a prompt to the API, creating a new revision only if needed.

If a prompt revision with the same normalized body and metadata already exists, the existing revision will be returned. If the metadata differs, a new revision will be created.

The engine parameter is only used to set property on output LoadedPrompt object. It is not persisted in any way and doesn't affect how the prompt is stored in Patronus AI Platform.

Note that when a new prompt definition is created, the description is used as provided. However, when creating a new revision for an existing prompt definition, the description parameter doesn't update the existing prompt definition's description.

Parameters:

| Name      | Type                                                             | Description                                                                                                 | Default     |
| --------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ----------- |
| `prompt`  | `Prompt`                                                         | The prompt to push                                                                                          | *required*  |
| `project` | `Union[str, Type[NOT_GIVEN]]`                                    | Optional project name override. If not specified, the project name from config is used.                     | `NOT_GIVEN` |
| `engine`  | `Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]]` | The template engine to use for rendering the returned prompt. If not specified, defaults to config setting. | `NOT_GIVEN` |

Returns:

| Name           | Type           | Description                             |
| -------------- | -------------- | --------------------------------------- |
| `LoadedPrompt` | `LoadedPrompt` | The created or existing prompt revision |

Raises:

| Type                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `PromptProviderError` | If there was an error communicating with the prompt provider. |

Source code in `src/patronus/prompts/clients.py`

```python
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
```

#### AsyncPromptClient

```python
AsyncPromptClient(provider_factory: Optional[ProviderFactory] = None)
```

Bases: `PromptClientMixin`

Source code in `src/patronus/prompts/clients.py`

```python
def __init__(self, provider_factory: Optional[ProviderFactory] = None) -> None:
    self._cache: AsyncPromptCache = AsyncPromptCache()
    self._provider_factory: ProviderFactory = provider_factory or {
        "local": lambda: LocalPromptProvider(),
        "api": lambda: APIPromptProvider(),
    }
    self._api_provider = APIPromptProvider()
```

##### get

```python
get(name: str, revision: Optional[int] = None, label: Optional[str] = None, project: Union[str, Type[NOT_GIVEN]] = NOT_GIVEN, disable_cache: bool = False, provider: Union[PromptProvider, _DefaultProviders, Sequence[Union[PromptProvider, _DefaultProviders]], Type[NOT_GIVEN]] = NOT_GIVEN, engine: Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]] = NOT_GIVEN) -> LoadedPrompt
```

Get the prompt asynchronously. If neither revision nor label is specified then the prompt with latest revision is returned.

Project is loaded from the config by default. You can specify the project name of the prompt if you want to override the value from the config.

By default, once a prompt is retrieved it's cached. You can disable caching.

Parameters:

| Name            | Type                                                                                                            | Description                                                                                                                                                                                  | Default     |
| --------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `name`          | `str`                                                                                                           | The name of the prompt to retrieve.                                                                                                                                                          | *required*  |
| `revision`      | `Optional[int]`                                                                                                 | Optional specific revision number to retrieve. If not specified, the latest revision is used.                                                                                                | `None`      |
| `label`         | `Optional[str]`                                                                                                 | Optional label to filter by. If specified, only prompts with this label will be returned.                                                                                                    | `None`      |
| `project`       | `Union[str, Type[NOT_GIVEN]]`                                                                                   | Optional project name override. If not specified, the project name from config is used.                                                                                                      | `NOT_GIVEN` |
| `disable_cache` | `bool`                                                                                                          | If True, bypasses the cache for both reading and writing.                                                                                                                                    | `False`     |
| `provider`      | `Union[PromptProvider, _DefaultProviders, Sequence[Union[PromptProvider, _DefaultProviders]], Type[NOT_GIVEN]]` | The provider(s) to use for retrieving prompts. Can be a string identifier ('local', 'api'), a PromptProvider instance, or a sequence of these. If not specified, defaults to config setting. | `NOT_GIVEN` |
| `engine`        | `Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]]`                                                | The template engine to use for rendering prompts. Can be a string identifier ('f-string', 'mustache', 'jinja2') or a TemplateEngine instance. If not specified, defaults to config setting.  | `NOT_GIVEN` |

Returns:

| Name           | Type           | Description                  |
| -------------- | -------------- | ---------------------------- |
| `LoadedPrompt` | `LoadedPrompt` | The retrieved prompt object. |

Raises:

| Type                  | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| `PromptNotFoundError` | If the prompt could not be found with the specified parameters. |
| `ValueError`          | If the provided provider or engine is invalid.                  |
| `PromptProviderError` | If there was an error communicating with the prompt provider.   |

Source code in `src/patronus/prompts/clients.py`

```python
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
```

##### push

```python
push(prompt: Prompt, project: Union[str, Type[NOT_GIVEN]] = NOT_GIVEN, engine: Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]] = NOT_GIVEN) -> LoadedPrompt
```

Push a prompt to the API asynchronously, creating a new revision only if needed.

If a prompt revision with the same normalized body and metadata already exists, the existing revision will be returned. If the metadata differs, a new revision will be created.

The engine parameter is only used to set property on output LoadedPrompt object. It is not persisted in any way and doesn't affect how the prompt is stored in Patronus AI Platform.

Note that when a new prompt definition is created, the description is used as provided. However, when creating a new revision for an existing prompt definition, the description parameter doesn't update the existing prompt definition's description.

Parameters:

| Name      | Type                                                             | Description                                                                                                 | Default     |
| --------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ----------- |
| `prompt`  | `Prompt`                                                         | The prompt to push                                                                                          | *required*  |
| `project` | `Union[str, Type[NOT_GIVEN]]`                                    | Optional project name override. If not specified, the project name from config is used.                     | `NOT_GIVEN` |
| `engine`  | `Union[TemplateEngine, DefaultTemplateEngines, Type[NOT_GIVEN]]` | The template engine to use for rendering the returned prompt. If not specified, defaults to config setting. | `NOT_GIVEN` |

Returns:

| Name           | Type           | Description                             |
| -------------- | -------------- | --------------------------------------- |
| `LoadedPrompt` | `LoadedPrompt` | The created or existing prompt revision |

Raises:

| Type                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `PromptProviderError` | If there was an error communicating with the prompt provider. |

Source code in `src/patronus/prompts/clients.py`

```python
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
```

### models

#### BasePrompt

##### with_engine

```python
with_engine(engine: Union[TemplateEngine, DefaultTemplateEngines]) -> typing.Self
```

Create a new prompt with the specified template engine.

Parameters:

| Name     | Type                                            | Description                                                                                | Default    |
| -------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------- |
| `engine` | `Union[TemplateEngine, DefaultTemplateEngines]` | Either a TemplateEngine instance or a string identifier ('f-string', 'mustache', 'jinja2') | *required* |

Returns:

| Type   | Description                                     |
| ------ | ----------------------------------------------- |
| `Self` | A new prompt instance with the specified engine |

Source code in `src/patronus/prompts/models.py`

```python
def with_engine(self, engine: Union[TemplateEngine, DefaultTemplateEngines]) -> typing.Self:
    """
    Create a new prompt with the specified template engine.

    Args:
        engine: Either a TemplateEngine instance or a string identifier ('f-string', 'mustache', 'jinja2')

    Returns:
        A new prompt instance with the specified engine
    """
    resolved_engine = get_template_engine(engine)
    return dataclasses.replace(self, _engine=resolved_engine)
```

##### render

```python
render(**kwargs: Any) -> str
```

Render the prompt template with the provided arguments.

If no engine is set on the prompt, the default engine from context/config will be used. If no arguments are provided, the template body is returned as-is.

Parameters:

| Name       | Type  | Description                                          | Default |
| ---------- | ----- | ---------------------------------------------------- | ------- |
| `**kwargs` | `Any` | Template arguments to be rendered in the prompt body | `{}`    |

Returns:

| Type  | Description         |
| ----- | ------------------- |
| `str` | The rendered prompt |

Source code in `src/patronus/prompts/models.py`

```python
def render(self, **kwargs: Any) -> str:
    """
    Render the prompt template with the provided arguments.

    If no engine is set on the prompt, the default engine from context/config will be used.
    If no arguments are provided, the template body is returned as-is.

    Args:
        **kwargs: Template arguments to be rendered in the prompt body

    Returns:
        The rendered prompt
    """
    if not kwargs:
        return self.body

    engine = self._engine
    if engine is None:
        # Get default engine from context
        engine_name = context.get_prompts_config().templating_engine
        engine = get_template_engine(engine_name)

    return engine.render(self.body, **kwargs)
```

#### calculate_normalized_body_hash

```python
calculate_normalized_body_hash(body: str) -> str
```

Calculate the SHA-256 hash of normalized prompt body.

Normalization is done by stripping whitespace from the start and end of the body.

Parameters:

| Name   | Type  | Description     | Default    |
| ------ | ----- | --------------- | ---------- |
| `body` | `str` | The prompt body | *required* |

Returns:

| Type  | Description                         |
| ----- | ----------------------------------- |
| `str` | SHA-256 hash of the normalized body |

Source code in `src/patronus/prompts/models.py`

```python
def calculate_normalized_body_hash(body: str) -> str:
    """Calculate the SHA-256 hash of normalized prompt body.

    Normalization is done by stripping whitespace from the start and end of the body.

    Args:
        body: The prompt body

    Returns:
        SHA-256 hash of the normalized body
    """
    normalized_body = body.strip()
    return hashlib.sha256(normalized_body.encode()).hexdigest()
```

### templating

#### TemplateEngine

Bases: `ABC`

##### render

```python
render(template: str, **kwargs) -> str
```

Render the template with the given arguments.

Source code in `src/patronus/prompts/templating.py`

```python
@abc.abstractmethod
def render(self, template: str, **kwargs) -> str:
    """Render the template with the given arguments."""
```

#### get_template_engine

```python
get_template_engine(engine: Union[TemplateEngine, DefaultTemplateEngines]) -> TemplateEngine
```

Convert a template engine name to an actual engine instance.

Parameters:

| Name     | Type                                            | Description                                                                                 | Default    |
| -------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------- | ---------- |
| `engine` | `Union[TemplateEngine, DefaultTemplateEngines]` | Either a template engine instance or a string identifier ('f-string', 'mustache', 'jinja2') | *required* |

Returns:

| Type             | Description                |
| ---------------- | -------------------------- |
| `TemplateEngine` | A template engine instance |

Raises:

| Type         | Description                                     |
| ------------ | ----------------------------------------------- |
| `ValueError` | If the provided engine string is not recognized |

Source code in `src/patronus/prompts/templating.py`

```python
def get_template_engine(engine: Union[TemplateEngine, DefaultTemplateEngines]) -> TemplateEngine:
    """
    Convert a template engine name to an actual engine instance.

    Args:
        engine: Either a template engine instance or a string identifier ('f-string', 'mustache', 'jinja2')

    Returns:
        A template engine instance

    Raises:
        ValueError: If the provided engine string is not recognized
    """
    if isinstance(engine, TemplateEngine):
        return engine

    if engine == "f-string":
        return FStringTemplateEngine()
    elif engine == "mustache":
        return MustacheTemplateEngine()
    elif engine == "jinja2":
        return Jinja2TemplateEngine()

    raise ValueError(
        "Provided engine must be an instance of TemplateEngine or "
        "one of the default engines ('f-string', 'mustache', 'jinja2'). "
        f"Instead got {engine!r}"
    )
```
