"""API-based prompt provider."""

import logging
from typing import Optional

import patronus_api

from patronus import context
from patronus.prompts.models import LoadedPrompt
from patronus.prompts.templating import TemplateEngine
from .base import PromptProvider, PromptProviderError

log = logging.getLogger("patronus.core")


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
