from typing import Literal, Any
from patronus.integrations.instrumenter import BasePatronusIntegrator

from opentelemetry.sdk._events import EventLoggerProvider


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from patronus import context


class PydanticAIIntegrator(BasePatronusIntegrator):
    """
    Integration for Pydantic-AI with Patronus.

    This class provides integration between Pydantic-AI agents and
    the Patronus observability stack, enabling tracing and logging
    of Pydantic-AI agent operations.
    """

    def __init__(self, event_mode: Literal["attributes", "logs"] = "logs"):
        """
        Initialize the Pydantic-AI integrator.

        Args:
            event_mode: The mode for capturing events, either as span attributes
                or as logs. Default is "logs".
        """
        self._instrumentation_settings = {"event_mode": event_mode}

    def apply(self, ctx: "context.PatronusContext", **kwargs: Any):
        """
        Apply Pydantic-AI instrumentation to the Patronus context.

        This method configures all Pydantic-AI agents to use the tracer and logger
        providers from the Patronus context.

        Args:
            ctx: The Patronus context containing the tracer and logger providers.
            **kwargs: Additional keyword arguments (unused).
        """
        from pydantic_ai.agent import Agent, InstrumentationSettings

        settings_kwargs = {
            **self._instrumentation_settings,
            "tracer_provider": ctx.tracer_provider,
            "event_logger_provider": EventLoggerProvider(ctx.logger_provider),
        }
        settings = InstrumentationSettings(**settings_kwargs)
        Agent.instrument_all(instrument=settings)
