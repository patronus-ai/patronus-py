"""Common utilities for prompt management commands."""

import sys
from typing import Optional, Dict, List

import patronus_api
from rich.console import Console

from patronus.config import config
from patronus.prompts.clients import PromptNotFoundError
from patronus.prompts.providers import (
    PromptProviderConnectionError,
    PromptProviderAuthenticationError,
)
from patronus.prompts.storage import LocalPromptStorage

console = Console()


def create_api_client(verbose: bool = False) -> patronus_api.Client:
    """Create and configure API client with proper error handling."""
    cfg = config()

    if verbose:
        console.print(f"Using API URL: {cfg.api_url}")

    if not cfg.api_key:
        raise PromptProviderAuthenticationError(
            "API key not found. Set PATRONUS_API_KEY environment variable or configure patronus.yaml"
        )

    client = patronus_api.Client(api_key=cfg.api_key, base_url=cfg.api_url, timeout=cfg.timeout_s)

    # Verify authentication
    client.whoami.retrieve()

    return client


def get_effective_resource_dir(resource_dir: Optional[str], ctx_resource_dir: Optional[str]) -> str:
    """Resolve the effective resource directory from multiple sources."""
    cfg = config()
    return resource_dir or ctx_resource_dir or cfg.resource_dir


def get_all_projects(client: patronus_api.Client, verbose: bool) -> List[str]:
    """Get all accessible projects."""
    try:
        if verbose:
            console.print("Fetching list of accessible projects...")

        resp = client.projects.list()
        projects = [p.name for p in resp.projects]

        if not projects:
            raise RuntimeError("No accessible projects found")

        return projects
    except Exception as e:
        raise PromptProviderConnectionError(f"Failed to fetch projects: {e}")


def get_prompt_definitions(
    client: patronus_api.Client,
    project_name: str,
    prompt_name: Optional[str],
    name_prefix: Optional[str],
    verbose: bool,
) -> List[Dict]:
    """Get prompt definitions for a project."""
    try:
        if prompt_name:
            # Get specific prompt
            resp = client.prompts.list_definitions(prompt_name=prompt_name, project_name=project_name)
        else:
            # Get all prompts in project
            resp = client.prompts.list_definitions(project_name=project_name)

        definitions = []
        for prompt_def in resp.prompt_definitions:
            # Apply name prefix filter if specified
            if name_prefix and not prompt_def.name.startswith(name_prefix):
                continue

            definitions.append(
                {
                    "id": prompt_def.id,
                    "name": prompt_def.name,
                    "description": prompt_def.description,
                    "project_id": prompt_def.project_id,
                }
            )

        return definitions

    except Exception as e:
        raise PromptProviderConnectionError(f"Failed to fetch prompt definitions for project {project_name}: {e}")


def update_storage_info(
    storage: LocalPromptStorage, project_name: str, project_id: str, prompt_name: str, revisions: List
) -> None:
    """Update storage info with project and prompt information."""
    # Update project info using the new helper method
    storage.update_project_info(project_name, project_id)

    # Update prompt labels using the new helper method
    labels = []
    for rev in revisions:
        for label_name in rev.labels:
            labels.append((label_name, rev.revision))

    storage.update_prompt_labels(project_name, prompt_name, labels)


def handle_command_error(e: Exception, verbose: bool) -> None:
    """Handle command errors with consistent formatting and exit codes."""
    if verbose:
        console.print_exception()
    else:
        console.print(f"[red]Error: {e}[/red]")

    # Set appropriate exit code based on error type
    if isinstance(e, PromptProviderAuthenticationError):
        sys.exit(3)  # Authentication error
    elif isinstance(e, PromptProviderConnectionError):
        sys.exit(1)  # Connection error
    elif isinstance(e, PromptNotFoundError):
        sys.exit(4)  # Not found error
    else:
        sys.exit(1)  # General error
