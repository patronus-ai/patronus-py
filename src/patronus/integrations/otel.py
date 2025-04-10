import typing
from patronus.integrations.instrumenter import BasePatronusIntegrator


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
    from patronus import context


class OpenTelemetryIntegrator(BasePatronusIntegrator):
    """
    Integration for OpenTelemetry instrumentors with Patronus.

    This class provides an adapter between OpenTelemetry instrumentors and
    the Patronus context, allowing for easy integration of OpenTelemetry
    instrumentation in Patronus-managed applications.
    """

    def __init__(self, instrumentor: "BaseInstrumentor"):
        """
        Initialize the OpenTelemetry integrator.

        Args:
            instrumentor: An OpenTelemetry instrumentor instance that will be
                applied to the Patronus context.
        """
        self.instrumentor = instrumentor

    def apply(self, ctx: "context.PatronusContext", **kwargs: typing.Any):
        """
        Apply OpenTelemetry instrumentation to the Patronus context.

        This method configures the OpenTelemetry instrumentor with the
        tracer provider from the Patronus context.

        Args:
            ctx: The Patronus context containing the tracer provider.
            **kwargs: Additional keyword arguments (unused).
        """
        self.instrumentor.instrument(tracer_provider=ctx.tracer_provider)
