import dataclasses
import datetime
import typing
from typing import Any, Optional

from patronus.prompts.templating import TemplateEngine, FStringTemplateEngine


class BasePrompt:
    name: str
    body: str
    description: Optional[str]
    metadata: Optional[str]

    _engine: TemplateEngine = FStringTemplateEngine()

    def with_engine(self, engine: TemplateEngine) -> typing.Self:
        return dataclasses.replace(self, _engine=engine)

    def render(self, **kwargs) -> str:
        if not kwargs:
            return self.body
        return self._engine.render(self.body, **kwargs)


@dataclasses.dataclass
class Prompt(BasePrompt):
    name: str
    body: str
    description: Optional[str] = None
    metadata: Optional[str] = None

    _engine: TemplateEngine = FStringTemplateEngine()


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

    _engine: TemplateEngine = FStringTemplateEngine()
