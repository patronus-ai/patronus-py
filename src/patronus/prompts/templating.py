import abc
from typing import Literal, Union


DefaultTemplateEngines = Literal["f-string", "mustache", "jinja2"]


class TemplateEngine(abc.ABC):
    @abc.abstractmethod
    def render(self, template: str, **kwargs) -> str:
        """Render the template with the given arguments."""


class FStringTemplateEngine(TemplateEngine):
    def render(self, template: str, **kwargs) -> str:
        return template.format(**kwargs)


class MustacheTemplateEngine(TemplateEngine):
    def __init__(self):
        try:
            import chevron

            self.chevron = chevron
        except ImportError:
            raise ImportError(
                "The 'chevron' package is required for MustacheTemplateEngine. "
                "Please install it with 'pip install chevron'."
            )

    def render(self, template: str, **kwargs) -> str:
        return self.chevron.render(template=template, data=kwargs)


class Jinja2TemplateEngine(TemplateEngine):
    def __init__(self):
        try:
            import jinja2

            self.jinja2 = jinja2
            self.environment = jinja2.Environment()
        except ImportError:
            raise ImportError(
                "The 'jinja2' package is required for Jinja2TemplateEngine. "
                "Please install it with 'pip install jinja2'."
            )

    def render(self, template: str, **kwargs) -> str:
        template_obj = self.environment.from_string(template)
        return template_obj.render(**kwargs)


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
