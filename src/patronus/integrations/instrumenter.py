import typing
import abc

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from patronus import context


class BasePatronusIntegrator(abc.ABC):
    """
    Abstract base class for Patronus integrations.

    This class defines the interface for integrating external libraries and tools
    with the Patronus context. All specific integrators should inherit from this
    class and implement the required methods.
    """

    @abc.abstractmethod
    def apply(self, ctx: "context.PatronusContext", **kwargs: typing.Any):
        """
        Apply the integration to the given Patronus context.

        This method must be implemented by subclasses to define how the
        integration is applied to a Patronus context instance.

        Args:
            ctx: The Patronus context to apply the integration to.
            **kwargs: Additional keyword arguments specific to the implementation.
        """
