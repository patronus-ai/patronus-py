"""
Context management for Patronus SDK.

This module provides classes and utility functions for managing the global Patronus
context and accessing different components of the SDK like logging, tracing, and API clients.
"""

import dataclasses
import logging
import pathlib

from opentelemetry import trace
from typing import Optional
from typing import TYPE_CHECKING

import patronus_api
from patronus.context.context_utils import ContextObject
from patronus.exceptions import UninitializedError

if TYPE_CHECKING:
    from patronus.evals.exporter import BatchEvaluationExporter
    from patronus.tracing.logger import Logger as PatLogger
    from patronus.api import PatronusAPIClient
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry._logs import LoggerProvider


@dataclasses.dataclass(frozen=True)
class PatronusScope:
    """
    Scope information for Patronus context.

    Defines the scope of the current Patronus application or experiment.

    Attributes:
        service: The service name as defined in OTeL.
        project_name: The project name.
        app: The application name.
        experiment_id: The unique identifier for the experiment.
        experiment_name: The name of the experiment.
    """

    service: Optional[str]
    project_name: Optional[str]
    app: Optional[str]
    experiment_id: Optional[str]
    experiment_name: Optional[str]


@dataclasses.dataclass(frozen=True)
class PromptsConfig:
    directory: pathlib.Path
    """The absolute path to a directory where prompts are stored locally."""
    providers: list[str]
    """List of default prompt providers."""
    templating_engine: str
    """Default prompt templating engine."""


@dataclasses.dataclass(frozen=True)
class PatronusContext:
    """
    Context object for Patronus SDK.

    Contains all the necessary components for the SDK to function properly.

    Attributes:
        scope: Scope information for this context.
        tracer_provider: The OpenTelemetry tracer provider.
        logger_provider: The OpenTelemetry logger provider.
        api_client_deprecated: Client for Patronus API communication (deprecated).
        api_client: Client for Patronus API communication using the modern client.
        async_api_client: Asynchronous client for Patronus API communication.
        exporter: Exporter for batch evaluation results.
        prompts: Configuration for prompt management.
    """

    scope: PatronusScope
    tracer_provider: "TracerProvider"
    logger_provider: "LoggerProvider"
    api_client_deprecated: "PatronusAPIClient"
    api_client: patronus_api.Client
    async_api_client: patronus_api.AsyncClient
    exporter: "BatchEvaluationExporter"
    prompts: "PromptsConfig"


_CTX_PAT = ContextObject[PatronusContext]("ctx.pat")


def set_global_patronus_context(ctx: PatronusContext):
    """
    Set the global Patronus context.

    Args:
        ctx: The Patronus context to set globally.
    """
    _CTX_PAT.set_global(ctx)


def get_current_context_or_none() -> Optional[PatronusContext]:
    """
    Get the current Patronus context or None if not initialized.

    Returns:
        The current PatronusContext if set, otherwise None.
    """
    return _CTX_PAT.get()


def get_current_context() -> PatronusContext:
    """
    Get the current Patronus context.

    Returns:
        The current PatronusContext.

    Raises:
        UninitializedError: If no active Patronus context is found.
    """
    ctx = get_current_context_or_none()
    if ctx is None:
        raise UninitializedError(
            "No active Patronus context found. Please initialize the library by calling patronus.init()."
        )
    return ctx


def get_logger(ctx: Optional[PatronusContext] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get a standard Python logger configured with the Patronus context.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.
        level: The logging level to set. Defaults to INFO.

    Returns:
        A configured Python logger.
    """
    from patronus.tracing.logger import set_logger_handler

    ctx = ctx or get_current_context()

    logger = logging.getLogger("patronus.sdk")
    set_logger_handler(logger, ctx.scope, ctx.logger_provider)
    logger.setLevel(level)
    return logger


def get_logger_or_none(level: int = logging.INFO) -> Optional[logging.Logger]:
    """
    Get a standard Python logger or None if context is not initialized.

    Args:
        level: The logging level to set. Defaults to INFO.

    Returns:
        A configured Python logger if context is available, otherwise None.
    """
    ctx = get_current_context()
    if ctx is None:
        return None
    return get_logger(ctx, level=level)


def get_pat_logger(ctx: Optional[PatronusContext] = None) -> "PatLogger":
    """
    Get a Patronus logger.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        A Patronus logger.
    """
    ctx = ctx or get_current_context()
    return ctx.logger_provider.get_logger("patronus.sdk")


def get_pat_logger_or_none() -> Optional["PatLogger"]:
    """
    Get a Patronus logger or None if context is not initialized.

    Returns:
        A Patronus logger if context is available, otherwise None.
    """
    ctx = get_current_context_or_none()
    if ctx is None:
        return None

    return ctx.logger_provider.get_logger("patronus.sdk")


def get_tracer(ctx: Optional[PatronusContext] = None) -> trace.Tracer:
    """
    Get an OpenTelemetry tracer.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        An OpenTelemetry tracer.
    """
    ctx = ctx or get_current_context()
    return ctx.tracer_provider.get_tracer("patronus.sdk")


def get_tracer_or_none() -> Optional[trace.Tracer]:
    """
    Get an OpenTelemetry tracer or None if context is not initialized.

    Returns:
        An OpenTelemetry tracer if context is available, otherwise None.
    """
    ctx = get_current_context_or_none()
    if ctx is None:
        return None
    return ctx.tracer_provider.get_tracer("patronus.sdk")


def get_api_client_deprecated(ctx: Optional[PatronusContext] = None) -> "PatronusAPIClient":
    """
    Get the Patronus API client.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The Patronus API client.
    """
    ctx = ctx or get_current_context()
    return ctx.api_client_deprecated


def get_api_client_deprecated_or_none() -> Optional["PatronusAPIClient"]:
    """
    Get the Patronus API client or None if context is not initialized.

    Returns:
        The Patronus API client if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.api_client_deprecated


def get_api_client(ctx: Optional[PatronusContext] = None) -> patronus_api.Client:
    """
    Get the Patronus API client.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The Patronus API client.
    """
    ctx = ctx or get_current_context()
    return ctx.api_client


def get_api_client_or_none() -> Optional[patronus_api.Client]:
    """
    Get the Patronus API client or None if context is not initialized.

    Returns:
        The Patronus API client if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.api_client


def get_async_api_client(ctx: Optional[PatronusContext] = None) -> patronus_api.AsyncClient:
    """
    Get the asynchronous Patronus API client.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The asynchronous Patronus API client.
    """
    ctx = ctx or get_current_context()
    return ctx.async_api_client


def get_async_api_client_or_none() -> Optional[patronus_api.AsyncClient]:
    """
    Get the asynchronous Patronus API client or None if context is not initialized.

    Returns:
        The asynchronous Patronus API client if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.async_api_client


def get_exporter(ctx: Optional[PatronusContext] = None) -> "BatchEvaluationExporter":
    """
    Get the batch evaluation exporter.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The batch evaluation exporter.
    """
    ctx = ctx or get_current_context()
    return ctx.exporter


def get_exporter_or_none() -> Optional["BatchEvaluationExporter"]:
    """
    Get the batch evaluation exporter or None if context is not initialized.

    Returns:
        The batch evaluation exporter if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.exporter


def get_scope(ctx: Optional[PatronusContext] = None) -> PatronusScope:
    """
    Get the Patronus scope.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The Patronus scope.
    """
    ctx = ctx or get_current_context()
    return ctx.scope


def get_scope_or_none() -> Optional[PatronusScope]:
    """
    Get the Patronus scope or None if context is not initialized.

    Returns:
        The Patronus scope if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.scope


def get_prompts_config(ctx: Optional[PatronusContext] = None) -> PromptsConfig:
    """
    Get the Patronus prompts configuration.

    Args:
        ctx: The Patronus context to use. If None, uses the current context.

    Returns:
        The Patronus prompts configuration.
    """
    ctx = ctx or get_current_context()
    return ctx.prompts


def get_prompts_config_or_none() -> Optional[PromptsConfig]:
    """
    Get the Patronus prompts configuration or None if context is not initialized.

    Returns:
        The Patronus prompts configuration if context is available, otherwise None.
    """
    return (ctx := get_current_context_or_none()) and ctx.prompts
