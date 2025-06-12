"""Prompt management commands.

This module provides CLI commands for managing prompts locally, including:

- Downloading prompts from the Patronus platform (`pull`)
- Cleaning up local prompt storage (`tidy`)
- Project-based prompt organization
- Label and revision management

All prompt commands integrate with the Patronus SDK's configuration system
and support both single-project and multi-project workflows.
"""

import click

from .pull import pull
from .tidy import tidy


@click.group()
def prompts():
    """Manage prompts locally.

    This command group provides functionality for downloading and managing
    prompts from the Patronus platform to local storage.

    Examples:

    ```bash
    # View available prompt commands
    patronus prompts --help

    # Download prompts from a project
    patronus prompts pull --project-name Global

    # Clean up local storage
    patronus prompts tidy
    ```
    """
    pass


# Add individual commands to the prompts group
prompts.add_command(pull)
prompts.add_command(tidy)
