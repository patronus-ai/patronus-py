"""Local file-based prompt provider."""

from typing import Optional

from patronus.prompts.models import LoadedPrompt
from patronus.prompts.templating import TemplateEngine
from .base import PromptProvider
from ..storage.storage import LocalPromptStorage


class LocalPromptProvider(PromptProvider):
    """Local prompt provider that reads from local storage."""

    def __init__(self, resource_dir: Optional[str] = None):
        if resource_dir is None:
            from patronus.config import config

            resource_dir = config().resource_dir
        self.storage = LocalPromptStorage(resource_dir)

    def get_prompt(
        self, name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine
    ) -> Optional[LoadedPrompt]:
        """Get a prompt from local storage."""
        # If specific revision requested, try to load it
        if revision is not None:
            prompt = self.storage.load_prompt_revision(project, name, revision)
            if prompt:
                prompt._engine = engine
                return prompt
            return None

        # If label requested, find the revision that has that label
        if label is not None:
            storage_info = self.storage.load_storage_info()
            for prompt_info in storage_info.prompts:
                if prompt_info.name == name and prompt_info.project_name == project:
                    for label_info in prompt_info.labels:
                        if label_info.name == label:
                            # Extract revision number from version string (e.g., "v4" -> 4)
                            try:
                                revision_num = int(label_info.points_to.version[1:])
                                prompt = self.storage.load_prompt_revision(project, name, revision_num)
                                if prompt:
                                    prompt._engine = engine
                                    return prompt
                            except (ValueError, IndexError):
                                continue
            return None

        # Default: return latest revision
        revisions = self.storage.list_available_revisions(project, name)
        if not revisions:
            return None

        latest_revision = max(revisions)
        prompt = self.storage.load_prompt_revision(project, name, latest_revision)
        if prompt:
            prompt._engine = engine
            return prompt

        return None

    async def aget_prompt(
        self, name: str, revision: Optional[int], label: Optional[str], project: str, engine: TemplateEngine
    ) -> Optional[LoadedPrompt]:
        """Async version - just calls the sync version."""
        return self.get_prompt(name, revision, label, project, engine)
