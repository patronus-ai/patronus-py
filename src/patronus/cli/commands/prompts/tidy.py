"""Tidy command for cleaning up local prompt storage.

This module implements the `patronus prompts tidy` command which cleans up
local prompt storage by removing outdated prompt revisions. It performs:

- Project validation (verifies projects still exist and IDs match)
- Label synchronization (updates local storage with current labels)
- Revision cleanup (removes non-latest, non-labeled revisions)
- Storage info updates (updates info.yaml to reflect changes)

The command helps maintain clean local storage by removing unnecessary
revisions while preserving latest and labeled revisions.
"""

from typing import Optional

import click

from .common import console, get_effective_resource_dir, handle_command_error


@click.command()
@click.option("--resource-dir", help="Override resource directory (global flag)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.pass_context
def tidy(ctx, resource_dir: Optional[str], verbose: bool) -> None:
    """Clean up local prompt storage by removing unlabeled, non-latest revisions.

    This command will:
    - Verify projects still exist and IDs match
    - Synchronize labels with the platform
    - Remove revisions that are neither latest nor labeled
    - Update info.yaml to reflect changes

    Examples:

    ```bash
    # Clean up current resource directory
    patronus prompts tidy

    # Clean up with verbose output
    patronus prompts tidy --verbose

    # Clean up custom resource directory
    patronus prompts tidy --resource-dir ./my-prompts
    ```
    """
    try:
        # Use resource_dir from global context if not specified locally
        effective_resource_dir = get_effective_resource_dir(resource_dir, ctx.obj.get("resource_dir"))

        _tidy_implementation(resource_dir=effective_resource_dir, verbose=verbose)
    except Exception as e:
        handle_command_error(e, verbose)


def _tidy_implementation(resource_dir: str, verbose: bool) -> None:
    """Implementation of the tidy command."""
    if verbose:
        console.print(f"Using resource directory: {resource_dir}")

    # TODO: Implement tidy functionality as per PRD
    # This includes:
    # 1. Project validation (verify projects still exist and IDs match)
    # 2. Label synchronization (query current labels from API)
    # 3. Revision cleanup (remove non-latest, non-labeled revisions)
    # 4. Storage info updates

    console.print("[red]Tidy command not yet implemented[/red]")

    if verbose:
        console.print("The tidy command will:")
        console.print("  - Verify projects still exist and IDs match")
        console.print("  - Synchronize labels with the platform")
        console.print("  - Remove revisions that are neither latest nor labeled")
        console.print("  - Update info.yaml to reflect changes")
