"""Base classes for prompt providers."""

import abc
from typing import Optional

from patronus.prompts.models import LoadedPrompt
from patronus.prompts.templating import TemplateEngine


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
