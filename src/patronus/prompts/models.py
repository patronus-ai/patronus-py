import dataclasses
import datetime
import hashlib
import typing
from typing import Any, Optional, Union

from patronus import context
from patronus.prompts.templating import TemplateEngine, DefaultTemplateEngines, get_template_engine


class BasePrompt:
    name: str
    body: str
    description: Optional[str]
    metadata: Optional[dict[str, Any]]

    _engine: Optional[TemplateEngine] = None

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


@dataclasses.dataclass
class Prompt(BasePrompt):
    name: str
    body: str
    description: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    _engine: Optional[TemplateEngine] = None


@dataclasses.dataclass
class LoadedPrompt(BasePrompt):
    prompt_definition_id: str
    project_id: str
    project_name: str

    name: str
    description: Optional[str]

    revision_id: str
    revision: int
    body: str
    normalized_body_sha256: str
    metadata: Optional[dict[str, Any]]
    labels: list[str]
    created_at: datetime.datetime

    _engine: Optional[TemplateEngine] = None


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
