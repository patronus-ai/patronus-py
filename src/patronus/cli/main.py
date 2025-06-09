"""Main CLI entry point for Patronus.

This module implements the main CLI interface for the Patronus SDK,
providing commands for prompt management, user authentication, and
project operations.

The CLI is built with Click and provides:
- Global options and configuration handling
- User authentication verification (whoami command)
- Prompt management command groups
- Consistent error handling and exit codes

All commands respect the same configuration hierarchy as the SDK:
1. Command-line arguments (highest priority)
2. Environment variables (PATRONUS_*)
3. Configuration file (patronus.yaml)
4. Default values (lowest priority)
"""

import json
import sys

import click
import patronus_api
from rich.console import Console

from patronus.config import config
from patronus.prompts.providers import (
    PromptProviderAuthenticationError,
    PromptProviderConnectionError,
)
from .commands.prompts import prompts

console = Console()


@click.group()
@click.version_option()
@click.option("--resource-dir", help="Override resource directory")
@click.pass_context
def main(ctx, resource_dir):
    """Patronus CLI for prompt management and more.

    The Patronus CLI provides command-line tools for managing prompts and
    interacting with the Patronus platform. It integrates seamlessly with
    the Patronus SDK configuration and authentication.

    Examples:

    ```bash
    # Check your authentication
    patronus whoami

    # Download prompts from a project
    patronus prompts pull --project-name Global

    # View all available commands
    patronus --help
    ```
    """
    ctx.ensure_object(dict)
    ctx.obj["resource_dir"] = resource_dir


@main.command()
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def whoami(verbose: bool) -> None:
    """Retrieve information about the current user and authentication status.

    This command verifies your API key and returns information about
    your user account and authentication status.

    Examples:

    ```bash
    # Check authentication status
    patronus whoami

    # Get detailed authentication information
    patronus whoami --verbose
    ```
    """
    try:
        # Get configuration
        cfg = config()

        if verbose:
            console.print(f"Using API URL: {cfg.api_url}")

        # Check for API key
        if not cfg.api_key:
            raise PromptProviderAuthenticationError(
                "API key not found. Set PATRONUS_API_KEY environment variable or configure patronus.yaml"
            )

        # Create API client
        client = patronus_api.Client(api_key=cfg.api_key, base_url=cfg.api_url, timeout=cfg.timeout_s)

        if verbose:
            console.print("Making whoami API call...")

        # Call whoami endpoint
        response = client.whoami.retrieve()

        # Convert response to JSON and output to stdout
        # Using model_dump() to convert Pydantic model to dict, then to JSON
        result_dict = response.model_dump()
        json_output = json.dumps(result_dict, indent=2, default=str)

        # Output to stdout (not using console.print to ensure clean JSON output)
        print(json_output)

    except PromptProviderAuthenticationError as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Authentication Error: {e}[/red]")
        sys.exit(3)  # Authentication error exit code
    except PromptProviderConnectionError as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Connection Error: {e}[/red]")
        sys.exit(1)  # General error exit code
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)  # General error exit code


# Add command groups
main.add_command(prompts)


if __name__ == "__main__":
    main()
